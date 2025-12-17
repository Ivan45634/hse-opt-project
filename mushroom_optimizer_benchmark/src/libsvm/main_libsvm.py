import numpy as np
import json
import wandb
import torch
from collections import defaultdict
import os
import optuna

from problems_libsvm import libsvm_prepocess
from optimizers.main import get_optimizer
from trainer import train
from utils import get_run_name

DATASETS = ["mushrooms", "binary"]
MAIN_METRIC = "accuracy"


def run_optimization(args, metrics, tuning=False, verbose=False):
    (train_dataloader, val_dataloader, test_dataloader, loss_fn, model) = (
        libsvm_prepocess(args)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = get_optimizer(args, model)

    _, val_results, test_results = train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_fn,
        device,
        args,
        tuning=tuning,
        verbose=verbose,
    )
    if hasattr(args, "save_curves") and args.save_curves and not tuning:
        curves = {
            "seed": int(args.seed),
            "run_name": args.run_name,
            "train_loss": val_results.get("train_loss", None),
            "val_accuracy": val_results.get("val_accuracy", None),
            "val_loss": val_results.get("val_loss", None),
            "test_accuracy": test_results.get("test_accuracy", None),
            "test_loss": test_results.get("test_loss", None),
        }
        out_path = f"./{args.results_path}/{args.run_name}__seed-{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(curves, f)

    if tuning:
        return np.max(val_results[f"val_{MAIN_METRIC}"])
    idx = np.argmax(val_results[f"val_{MAIN_METRIC}"])
    for metric in test_results:
        metrics[metric].append(test_results[metric][idx])
    
    return metrics


def tune_params(args, parser, use_old_tune_params=True):
    tune_name = get_run_name(args, parser, tuning=True)
    f_name = f"{args.tune_path}/{tune_name}.json"
    if os.path.exists(f_name) and use_old_tune_params:
        try:
            with open(f_name) as f:
                params = json.load(f)
                for key in params.keys():
                    setattr(args, key, params[key])
            return args, params.keys()
        except json.decoder.JSONDecodeError:
            pass

    study = optuna.create_study(direction="maximize", study_name=f"{tune_name}")

    def tune_function(trial):
        args.lr = trial.suggest_float("lr", 1e-6, 5e0, log=True)
        if hasattr(args, "weight_decay"):
            args.weight_decay = trial.suggest_float(
                "weight_decay", 1e-6, 1e-2, log=True
            )
        # if hasattr(args, "momentum"):
        #     args.momentum = trial.suggest_float("momentum", 1e-6, 1., log=True)
        # if hasattr(args, "beta1"):
        #     args.beta1 = trial.suggest_float("beta1", 1e-6, 1., log=True)
        # if hasattr(args, "beta2"):
        #     args.beta2 = trial.suggest_float("beta2", 1e-6, 1., log=True)

        return run_optimization(args, None, tuning=True, verbose=False)

    study.optimize(tune_function, n_trials=args.tune_runs)
    with open(f_name, "w") as f:
        json.dump(study.best_trial.params, f)

    with open(f_name) as f:
        params = json.load(f)
        for key in params.keys():
            setattr(args, key, params[key])

    return args, params.keys()


def main(args, parser):
    args.results_path = f"./src/libsvm/{args.results_path}"
    args.tune_path = f"./src/libsvm/{args.tune_path}"
    os.makedirs(f"./{args.results_path}", exist_ok=True)
    os.makedirs(f"./{args.tune_path}", exist_ok=True)
    os.makedirs(f"./{args.data_path}", exist_ok=True)

    torch.set_default_dtype(getattr(torch, args.dtype))
    metrics = defaultdict(list)

    if args.tune or args.use_old_tune_params:
        if args.use_old_tune_params is False:
            print("~~~~~~~~~~~~~~~ TUNING ~~~~~~~~~~~~~~~")
        args, tuned_params = tune_params(
            args, parser, use_old_tune_params=args.use_old_tune_params
        )
        if len(tuned_params) > 0:
            print("~~~~~~~~~~~~~~~ TUNED PARAMS ~~~~~~~~~~~~~~~")
            for param_name in tuned_params:
                print(f"{param_name:<20} {getattr(args, param_name)}")

    for i, seed in enumerate(range(args.eval_runs)):
        # args.seed = 52 * 52 * 52 + seed
        args.seed = seed
        if args.wandb:
            wandb.init(
                project=args.wandb_project,
                tags=[args.model, args.dataset, args.optimizer],
                name=args.run_name,
                config=args,
            )

        print(f"~~~~~~~~~~~~~~~ TRAIN RUN {i+1}/{args.eval_runs} ~~~~~~~~~~~~~~~")
        metrics = run_optimization(args, metrics, tuning=False, verbose=args.verbose)
        if args.wandb:
            wandb.finish()

    with open(f"./{args.results_path}/{args.run_name}.txt", "w") as f:
        f.write(f"~~~~~~~~~~~~~~~ Arguments ~~~~~~~~~~~~~~~\n")
        for key, value in vars(args).items():
            f.write(f"{key:<20} {value}\n")
        f.write(f"~~~~~~~~~~~~~~~ Results ~~~~~~~~~~~~~~~\n")
        for metric_name in metrics.keys():
            res = np.array(metrics[metric_name])
            f.write(f"{metric_name}: {res.mean()}+-{res.std()}\n")

    print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
