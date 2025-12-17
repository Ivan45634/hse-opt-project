def set_arguments_libsvm(parser):
    ### Dataset Arguments
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Use or not scaling for the libsvm datasets",
    )
    parser.add_argument(
        "--scale_bound",
        default=3.0,
        type=float,
        help="Scaling ~`exp[U(-scale_bound, scale_bound)]`",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Use or not rotating for the libsvm datasets",
    )

    ### Model Arguments
    parser.add_argument(
        "--model",
        default="linear-classifier",
        choices=["linear-classifier", "mlp"],
        help="Model name",
    )
    parser.add_argument(
        "--hidden_dim",
        default=10,
        type=int,
        help="Hidden dimatially of linear classifier",
    )
    parser.add_argument(
        "--no_bias",
        action="store_true",
        help="No bias in the FCL of the linear classifier",
    )
    parser.add_argument(
        "--weight_init",
        default="uniform",
        choices=["zeroes", "uniform", "bad_scaled", "ones", "zero/uniform"],
        help="Initial weights of the linear classifier",
    )

    ### Training Arguments (change defaults)
    parser.set_defaults(batch_size=128, n_epoches_train=2, eval_runs=3, dtype="float64")

    ### Tuning Arguments
    parser.add_argument("--tune", action="store_true", help="Tune params")
    parser.add_argument(
        "--n_epoches_tune",
        default=1,
        type=int,
        help="How many epochs to tune with optuna",
    )
    parser.add_argument(
        "--tune_runs", default=20, type=int, help="Number of optuna steps"
    )
    parser.add_argument(
        "--tune_path", default="tuned_params", help="Path to save the tuned params"
    )
    parser.add_argument(
        "--save_curves", action="store_true",
        help="Save per-epoch curves (train/val/test) into results_path as JSON"
    )
    
    return parser
