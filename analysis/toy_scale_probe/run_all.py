from exp1_static_order import run_experiment_1
from exp2_gamma_logit import run_experiment_2
from exp3_gamma_logit import run_experiment_3


def main() -> None:
    exp1_summary, exp1_figures = run_experiment_1()
    exp2_summary, _ = run_experiment_2()
    exp3_summary, _ = run_experiment_3()

    print("exp1_summary (in memory only)")
    print(exp1_summary.to_string(index=False))
    print()
    print("exp1_figures")
    for setting_key, output_path in sorted(exp1_figures.items()):
        print(f"{setting_key}: {output_path}")
    print()
    print("exp2_summary.csv")
    print(exp2_summary.to_string(index=False))
    print()
    print("exp3_summary (in memory only)")
    print(exp3_summary.to_string(index=False))


if __name__ == "__main__":
    main()
