import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os
import numpy as np
from sklearn.datasets import load_svmlight_file, make_classification

import utils
from libsvm import models_libsvm

def libsvm_prepocess(args):
    g, seed_worker = utils.set_global_seed(args.seed)
    if args.dataset == "mushrooms":
        if not os.path.exists(f"./{args.data_path}/mushrooms"):
            os.system(f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd ..")
        X, y = load_svmlight_file(f"./{args.data_path}/mushrooms")
        y = y - 1
        X = X.toarray()
    if args.dataset == "binary":
        if not os.path.exists(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2"):
            os.system(f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2 \n cd ..")
        X, y = load_svmlight_file(f"./{args.data_path}/covtype.libsvm.binary.scale.bz2")
        y = y - 1
        X = X.toarray()
    
    if args.scale:
        A = np.diag(np.exp(np.random.uniform(-args.scale_bound, args.scale_bound, X.shape[1])))
        X = X @ A
    if args.rotate:
        B = np.random.random([X.shape[1], X.shape[1]])
        A, _ = np.linalg.qr(B.T @ B)
        X = X @ A
    
    X = torch.tensor(X)
    y = torch.tensor(y, dtype=X.dtype)

    ds = TensorDataset(X, y)
    ds_train, ds_val, ds_test = torch.utils.data.random_split(ds, [0.7, 0.2, 0.1],
                                                              generator=g)
    train_dataloader = DataLoader(
        ds_train, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        ds_val, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        ds_test, 
        batch_size=args.batch_size,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if args.model == "linear-classifier":
        model = models_libsvm.LinearClassifier(
            input_dim=X.shape[1], hidden_dim=args.hidden_dim, 
            output_dim=len(np.unique(y)), dtype=X.dtype, bias=not args.no_bias,
            weight_init=args.weight_init
        )
    else:
        raise ValueError(f"Wrong model name: {args.model} for dataset {args.dataset}")

    return train_dataloader, val_dataloader, test_dataloader, loss_fn, model
