# QK-MAP

This repository contains the main implementation and experiment code for
QK-MAP. It is intentionally source-only. It does **not** include datasets, model
checkpoints, logs, generated results, W&B files, caches, machine-specific
launchers, credentials, or absolute filesystem paths.

## Contents

- `qkmap/core.py`: the main magnitude-aware periodic map and Q/K/V projection.
- `qkmap/olmo_integration.py`: the changed projection step for an OLMo-style
  attention block.
- `training/pretrain_olmo.py`: archived OLMo-core training configuration
  scaffolding for five model scales; integration is required (see below).
- `analysis/toy_scale_probe/`: the training-free magnitude-response probes.
- `analysis/checkpoint_scale_response/`: the checkpoint-based scale-response
  analysis.
- `analysis/norm_shuffle/`: the causal query/key norm-shuffling intervention.
- `analysis/parse_sink_ratio.py`: the attention-sink report parser.
- `analysis/summarize_niah.py`: the needle-attention summary script.
- `evaluation/olmes/`: OLMES result aggregation and paired bootstrap code.
- `tests/`: small checks for the main QK-MAP properties.

## Method implementation

For an input hidden state `x`, the main same-width map is

```text
z = x W_p + b_p
Phi(x) = concat(cos(z), sin(z))
q = W_Q Phi(x)
k = W_K Phi(x)
v = W_V x
```

The main setting uses a periodic ratio of `0.5`, so `z` has half the hidden
width and `Phi(x)` has the original hidden width. The query and key paths
share the same map. The value path is unchanged.

`qkmap/core.py` also contains the residual variants used in the architecture
ablation. The paper's parameter-matched residual variant uses
`residual="learnable_per_dim"` and initializes its residual coefficient to
zero.

## Environment

The core module and analysis code require Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The training configuration additionally requires OLMo-core and its normal
training dependencies. The experiments were based on OLMo-core commit
`697f315ed17422ee0bc57ee9244eb312c4a4b0b8`. Because OLMo-core is an external
project, it is not copied into this archive.

## OLMo integration

Inside each attention block:

1. Build one `MagnitudeAwareProjection(d_model, d_model)` shared by Q and K.
2. Replace the original Q/K projection input with
   `mapped_qk_input(x, qkmap, ...)`.
3. Keep the V projection input as `x`.
4. Reshape Q/K into heads and apply RoPE in the usual position.

The exact changed projection step is implemented by
`qkmap.olmo_integration.project_qkv`. All attention backends, RoPE, masking,
output projection, and tensor-parallel logic remain unchanged.

The archived training script does not automatically insert the QK-MAP helpers
into an unmodified OLMo-core model. Integrate `project_qkv` and register the map
and residual parameters in each attention block before using it for QK-MAP
training. The script also relies on model presets and callbacks from the
experiment's OLMo-core environment; this package is not a standalone copy of
that environment. Use the architecture and feed-forward widths in the paper's
appendix for parameter matching.

Storage and dataset locations are supplied through command-line arguments.
After integration, a configuration example is:

```bash
torchrun --nproc-per-node=8 training/pretrain_olmo.py \
  --model-size=190M \
  --name=qkmap-190m \
  --save-folder=OUTPUT_DIR \
  --data-root=DATASET_DIR \
  --lr-scheduler=cosine \
  --global-batch-size=2097152 \
  --disable-wandb
```

The model-size defaults match the paper: 2B, 4B, 8B, 10B, and 20B training
tokens for 190M, 370M, 760M, 1B, and 7B models, respectively. Dataset and
output directories are deliberately left to the user.

The explicit scheduler and batch-size overrides above match the paper. The
archived script also retains defaults from other runs (a midpoint restart and
a larger batch for 1B/7B); do not use those defaults for the reported comparison.
Later QUEST, UEA, and word-replacement experiments are not included in this
core-code package.

## Training-free probe

The scale probe generates its own synthetic inputs and does not require a
dataset or checkpoint:

```bash
cd analysis/toy_scale_probe
python run_all.py
```

Settings, seeds, and output location are defined in `config.py`.

## Checkpoint analyses

Checkpoint and dataset paths are explicit command-line inputs. For example:

```bash
python analysis/checkpoint_scale_response/analyze_scale_response.py \
  --output-dir=OUTPUT_DIR \
  --model="Standard=STANDARD_CHECKPOINT" \
  --model="QK-Norm=QKNORM_CHECKPOINT" \
  --model="QK-MAP=QKMAP_CHECKPOINT"
```

```bash
python analysis/norm_shuffle/run_norm_shuffle.py \
  --output-dir=OUTPUT_DIR \
  --data=TOKENIZED_VALIDATION_NPY \
  --model="Standard=STANDARD_CHECKPOINT" \
  --model="QK-Norm=QKNORM_CHECKPOINT" \
  --model="QK-MAP=QKMAP_CHECKPOINT"
```

These commands write new result files to `OUTPUT_DIR`; no result files are
included in this archive.

## OLMES aggregation

```bash
python evaluation/olmes/summarize_olmes.py \
  --run="Standard=STANDARD_METRICS_ALL_JSONL" \
  --run="QK-Norm=QKNORM_METRICS_ALL_JSONL" \
  --run="QK-MAP=QKMAP_METRICS_ALL_JSONL" \
  --output=OUTPUT_CSV
```

The script uses the primary metric recorded by each OLMES task and checks that
all compared runs use the same evaluation protocol. The paired bootstrap
script consumes per-example prediction JSONL files.

## Data and checkpoints

No dataset or checkpoint is redistributed. The pretraining experiments use
the OLMo data pipeline and Dolma-family data described in the paper. OLMES
evaluation uses the public task definitions provided by the OLMES project.
