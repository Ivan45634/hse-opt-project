# scripts/libsvm/run_mushrooms_table1.py
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRY = REPO_ROOT / "src" / "run_experiment.py"

METHODS = [
    # Norm-based
    "normalized_sgd",
    "signsgd",
    # "muon",
    # "scion",
    # Quasi-Newton / Kronecker
    "kfac",
    # "shampoo",
    # "onesided_shampoo",
    # "kl_shampoo",
    # Adaptive
    # "adagrad",
    "adam",
    "madgrad",
    # "adam_sania",
    # Hybrid
    "soap",
    "splus",
    "muadam",
    # "muadam_sania",
]

def run(cmd):
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

def main():
    # IMPORTANT: disable optuna tuning for new optimizers
    COMMON = [
        "python", str(ENTRY),
        "--dataset", "mushrooms",
        "--config_name", "basic",
        "--results_path", "results_raw",
        "--eval_runs", "3",
        "--n_epoches_train", "200",
        "--batch_size", "256",
        "--model", "linear-classifier",
        "--hidden_dim", "10",
        "--no_bias",
        "--weight_init", "uniform",
        "--verbose",
        "--save_curves",
        # do NOT set --use_old_tune_params unless you have tuned files for all methods
    ]

    # For reproducibility: run 3 different seeds by invoking runs separately
    SEEDS = [18, 19, 20]

    for scale_flag in [False, True]:
        for method in METHODS:
            for seed in SEEDS:
                run_prefix = f"tbl1_{method}_{'scaled' if scale_flag else 'orig'}"
                cmd = COMMON + [
                    "--optimizer", method,
                    "--seed", str(seed),
                    "--run_prefix", run_prefix,
                ]
                if scale_flag:
                    cmd += ["--scale"]

                # reasonable defaults
                if method in ("sgd", "normalized_sgd", "signsgd", "scion", "muon", "muadam", "muadam_sania"):
                    cmd += ["--momentum", "0.9"]

                if method in ("muon", "scion", "muadam", "muadam_sania"):
                    cmd += ["--ns_steps", "20"]

                if method in ("soap", "splus", "kfac", "shampoo", "onesided_shampoo", "kl_shampoo"):
                    cmd += ["--update_freq", "1"]

                # EMA beta: for kfac/kl_shampoo use EMA; for shampoo you can keep -1 for sum if you want
                if method in ("kfac", "kl_shampoo", "soap", "splus"):
                    cmd += ["--shampoo_beta", "0.9"]
                if method in ("shampoo", "onesided_shampoo"):
                    cmd += ["--shampoo_beta", "-1"]

                run(cmd)

if __name__ == "__main__":
    main()
