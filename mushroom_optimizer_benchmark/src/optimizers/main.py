import torch.optim as optim
import sys

# from torch_optimizer import Shampoo
sys.path.append("src/optimizers")
import soap, muon
import taia
import optimizer_classes as oc


def get_optimizer(args, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "soap":
        optimizer = soap.SOAP(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_freq,
            max_precond_dim=args.max_precond_dim,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "muon":
        optimizer = muon.Muon(
            muon_params=trainable_params,
            lr=args.lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
        )
    elif args.optimizer == "taia":
        optimizer = taia.TAIA(
            taia_params=trainable_params,
            lr=args.lr,
            momentum=args.momentum,
            ns_steps=args.ns_steps,
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            lmo=args.lmo,
            precondition_type=args.precondition_type,
        )
    # ------ Additional optimizers --------
    elif args.optimizer in ("normalized_sgd", "norm_sgd"):
        optimizer = oc.NormalizedSGD(
            params=trainable_params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )
    elif args.optimizer == "signsgd":
        optimizer = oc.SignSGD(
            params=trainable_params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "scion":
        optimizer = oc.Scion(
            params=trainable_params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            weight_decay=args.weight_decay,
            ns_steps=getattr(args, "ns_steps", 10),
            eps=args.eps,
            max_rows=10000,
        )
    elif args.optimizer == "muadam":
        optimizer = taia.TAIA(
            taia_params=trainable_params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            ns_steps=getattr(args, "ns_steps", 10),
            adamw_lr=getattr(args, "adamw_lr", args.lr),
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            lmo="spectral",
            precondition_type="adam",
        )
    elif args.optimizer in ("muadam_sania", "muadam-sania"):
        optimizer = taia.TAIA(
            taia_params=trainable_params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.0),
            ns_steps=getattr(args, "ns_steps", 10),
            adamw_lr=getattr(args, "adamw_lr", args.lr),
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            adamw_wd=args.weight_decay,
            lmo="spectral",
            precondition_type="adam_sania",
        )
    elif args.optimizer == "adagrad":
        optimizer = oc.AdaGradTable1(
            params=trainable_params,
            lr=args.lr,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = oc.AdamTable1(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "madgrad":
        optimizer = oc.MADGRADTable1(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer in ("adam_sania", "adam-sania"):
        optimizer = oc.AdamSANIA(
            params=trainable_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "splus":
        optimizer = oc.SPlusTable1(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            update_freq=args.update_freq,
            beta=args.shampoo_beta,
            max_rows=10000,
            fallback_adam=True,
            adam_betas=(args.beta1, args.beta2),
        )
    # -------- Table 1 (quasi-Newton / Kronecker) --------
    elif args.optimizer == "kfac":
        optimizer = oc.KFACTable1(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            update_freq=args.update_freq,
            beta=max(0.0, float(args.shampoo_beta)) if args.shampoo_beta is not None else 0.9,
            max_rows=10000,
            fallback_adam=True,
            adam_betas=(args.beta1, args.beta2),
        )
    elif args.optimizer == "shampoo":
        optimizer = oc.ShampooTable1(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            update_freq=args.update_freq,
            beta=args.shampoo_beta,   # beta<0 => sum; beta>=0 => EMA
            max_rows=10000,
            fallback_adam=True,
            adam_betas=(args.beta1, args.beta2),
        )
    elif args.optimizer in ("onesided_shampoo", "one_sided_shampoo"):
        optimizer = oc.OneSidedShampooTable1(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            update_freq=args.update_freq,
            beta=args.shampoo_beta,
            max_rows=10000,
            fallback_adam=True,
            adam_betas=(args.beta1, args.beta2),
        )
    elif args.optimizer in ("kl_shampoo", "kl-shampoo"):
        optimizer = oc.KLShampooTable1(
            params=trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            update_freq=args.update_freq,
            beta=args.shampoo_beta,
            max_rows=10000,
            fallback_adam=True,
            adam_betas=(args.beta1, args.beta2),
        )
    else:
        raise NotImplementedError(f"Wrong optimizer name {args.optimizer}")
    return optimizer
