"""OLMo-core pretraining entrypoint used for the QK-MAP experiments.

All storage and dataset locations are supplied through command-line arguments.
The script contains no machine-specific paths or credentials.
"""

import argparse
import os
from typing import List

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    CosWithWarmupAndMidpointRestart,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.script_utils import ExperimentConfig, get_cli_parser, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    LMEvaluatorCallbackConfig,
    MetricSaverCallback,
    MonkeyPatcherCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

DEFAULT_SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE_1B_7B = 2048 * DEFAULT_SEQUENCE_LENGTH
GLOBAL_BATCH_SIZE_DEFAULT = 1024 * DEFAULT_SEQUENCE_LENGTH
LR_DEFAULT = 6e-4
LR_1B = 4e-4
LR_7B = 3e-4
DEFAULT_WANDB_PROJECT = "qkmap"

DEFAULT_MAX_DURATION_B = {
    "190M": 2.0,
    "370M": 4.0,
    "760M": 8.0,
    "1B": 10.0,
    "7B": 20.0,
}

DATA_LOADER_SEED = 34521


def _get_model_config(size: str, vocab_size: int) -> TransformerConfig:
    common = dict(
        attn_backend=AttentionBackendName.flash_2,
        block_name=TransformerBlockType.default,
    )
    if size == "190M":
        return TransformerConfig.olmo2_190M(vocab_size=vocab_size, **common)
    if size == "370M":
        return TransformerConfig.olmo2_370M(vocab_size=vocab_size, **common)
    if size == "760M":
        return TransformerConfig.olmo2_760M(vocab_size=vocab_size, **common)
    if size == "1B":
        return TransformerConfig.olmo2_1B(vocab_size=vocab_size, **common)
    if size == "7B":
        return TransformerConfig.olmo2_7B(vocab_size=vocab_size, **common)
    raise ValueError(f"Unknown model-size: {size}")


def _get_learning_rate(model_size: str, override: float | None) -> float:
    if override is not None:
        return override
    if model_size == "7B":
        return LR_7B
    if model_size == "1B":
        return LR_1B
    return LR_DEFAULT


def _get_global_batch_size(model_size: str, sequence_length: int, override: int | None) -> int:
    del sequence_length
    if override is not None:
        return override
    if model_size in ("1B", "7B"):
        return GLOBAL_BATCH_SIZE_1B_7B
    return GLOBAL_BATCH_SIZE_DEFAULT


def _build_scheduler(opts: argparse.Namespace):
    if opts.lr_scheduler == "cosine":
        return CosWithWarmup(
            warmup_fraction=opts.lr_scheduler_warmup_fraction,
            alpha_f=opts.lr_scheduler_alpha_f,
        )
    if opts.lr_scheduler == "midpoint_restart":
        return CosWithWarmupAndMidpointRestart(
            warmup_fraction=opts.lr_scheduler_warmup_fraction,
            alpha_f=opts.lr_scheduler_alpha_f,
            restart_fraction=opts.lr_scheduler_restart_fraction,
        )
    raise ValueError(f"Unknown lr scheduler: {opts.lr_scheduler}")


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    sequence_length = getattr(opts, "sequence_length", None) or DEFAULT_SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.dolma2()
    model_size = opts.model_size

    model_config = _get_model_config(model_size, tokenizer_config.padded_vocab_size())

    dataset_config = NumpyFSLDatasetConfig.from_data_mix(
        DataMix.OLMo_mix_0625_20Bsample,
        tokenizer=tokenizer_config,
        mix_base_dir=opts.data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(8192, sequence_length),
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=_get_global_batch_size(
            model_size,
            sequence_length,
            getattr(opts, "global_batch_size", None),
        ),
        seed=DATA_LOADER_SEED,
        num_workers=8,
    )

    lr = _get_learning_rate(model_size, getattr(opts, "learning_rate", None))
    rank_microbatch_size = getattr(opts, "rank_microbatch_size", None) or (8 * sequence_length)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=_build_scheduler(opts),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    run_name = opts.name or f"qkmap-{model_size}-{opts.lr_scheduler}"
    save_folder = os.path.join(opts.save_folder, run_name)
    max_duration_b = (
        opts.max_duration_b
        if opts.max_duration_b is not None
        else DEFAULT_MAX_DURATION_B.get(model_size, 8.0)
    )
    max_tokens = int(max_duration_b * 1e9)

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(max_tokens),
        )
        .with_callback("monkey_patcher", MonkeyPatcherCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=None,
                save_async=False,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project=getattr(opts, "wandb_project", None) or DEFAULT_WANDB_PROJECT,
                entity=getattr(opts, "wandb_entity", None),
                cancel_check_interval=10,
                enabled=not opts.disable_wandb,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "metric_saver",
            MetricSaverCallback(
                final_metrics_fname="metrics.json",
                save_interval=500,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=os.path.dirname(opts.data_root),
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=100,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=sorted(
                    [
                        "arc_challenge",
                        "arc_easy",
                        "basic_arithmetic",
                        "boolq",
                        "commonsense_qa",
                        "copa",
                        "gsm8k_gold_bpb_5shot",
                        "hellaswag",
                        "mmlu_humanities_mc_5shot",
                        "mmlu_other_mc_5shot",
                        "mmlu_social_sciences_mc_5shot",
                        "mmlu_stem_mc_5shot",
                        "openbook_qa",
                        "piqa",
                        "sciq",
                        "social_iqa",
                        "winogrande",
                    ]
                ),
                tokenizer=tokenizer_config,
                eval_interval=100,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-size",
        type=str,
        default="190M",
        choices=["190M", "370M", "760M", "1B", "7B"],
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--rank-microbatch-size", type=int, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B logging. No API key is stored in this repository.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="midpoint_restart",
        choices=["cosine", "midpoint_restart"],
    )
    parser.add_argument(
        "--lr-scheduler-alpha-f",
        type=float,
        default=0.1,
        help="Final LR ratio relative to max LR.",
    )
    parser.add_argument(
        "--lr-scheduler-warmup-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lr-scheduler-restart-fraction",
        type=float,
        default=0.5,
        help="Restart point for the midpoint-restart scheduler.",
    )


if __name__ == "__main__":
    parser = get_cli_parser()
    _add_args(parser)
    main(build_config, parser=parser)
