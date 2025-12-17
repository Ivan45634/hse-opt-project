# src/optimizers/optimizer_classes.py
import math
from typing import Optional, Tuple

import torch
from torch.optim import Optimizer

import lmo


# ----------------------------
# Utilities
# ----------------------------

@torch.no_grad()
def _sym_matrix_power(A: torch.Tensor, power: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric matrix power via eigen-decomposition.
    A must be symmetric PSD (we enforce symmetry numerically).
    """
    A = 0.5 * (A + A.T)
    w, Q = torch.linalg.eigh(A)
    w = torch.clamp(w, min=eps)
    wp = w.pow(power)
    return (Q * wp.unsqueeze(0)) @ Q.T


@torch.no_grad()
def _inv_root_psd(A: torch.Tensor, root: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns A^{-1/root}, for PSD A, via eigen-decomposition.
    Example: root=4 -> A^{-1/4}
    """
    return _sym_matrix_power(A, power=-1.0 / root, eps=eps)


@torch.no_grad()
def _as_matrix(G: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Flatten >=2D tensors to 2D (first dim preserved) to support matrix preconditioning.
    """
    if G.ndim == 2:
        return G, G.shape
    if G.ndim >= 3:
        shp = G.shape
        return G.view(shp[0], -1), shp
    # 1D stays 1D
    return G, G.shape


@torch.no_grad()
def _restore_shape(X: torch.Tensor, shp: Tuple[int, ...]) -> torch.Tensor:
    if X.shape == shp:
        return X
    return X.view(shp)


def _use_matrix_rules(p: torch.Tensor, max_rows: int = 10000) -> bool:
    return (p.ndim >= 2) and (p.shape[0] < max_rows)


# ----------------------------
# Norm-based methods
# ----------------------------

class NormalizedSGD(Optimizer):
    """
    Normalized SGD (Frobenius base norm).
    Update direction is -g / ||g|| (per-parameter tensor).
    Optional momentum.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, eps=1e-12):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if mu != 0.0:
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                    buf.mul_(mu).add_(g)
                    g_eff = buf
                else:
                    g_eff = g

                # normalized direction
                direction = -g_eff / (torch.norm(g_eff) + eps)
                p.add_(direction, alpha=lr)

        return loss


class SignSGD(Optimizer):
    """
    SignSGD (l1 -> linf base norm in practice).
    Optional momentum on raw gradients.
    """
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if mu != 0.0:
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                    buf.mul_(mu).add_(g)
                    g_eff = buf
                else:
                    g_eff = g

                p.add_(torch.sign(g_eff), alpha=-lr)

        return loss


class Scion(Optimizer):
    """
    Scion (Table 1): matrices use Spectral base norm; vectors use l_infty (implemented as sign step).
    We implement:
      - For matrices (ndim>=2): spectral LMO via Newton–Schulz.
      - For vectors (ndim==1): sign step.
    Optional momentum.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.0,
        weight_decay=0.0,
        ns_steps: int = 10,
        eps: float = 1e-7,
        max_rows: int = 10000,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, eps=eps, max_rows=max_rows
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            max_rows = group["max_rows"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if mu != 0.0:
                    buf = state.get("momentum_buffer", None)
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.zeros_like(g)
                    buf.mul_(mu).add_(g)
                    g_eff = buf
                else:
                    g_eff = g

                if _use_matrix_rules(p, max_rows=max_rows):
                    Gm, shp = _as_matrix(g_eff)
                    direction = lmo.spectral_lmo(Gm, ns_steps=ns_steps, eps=eps, scale_rectangular=True)
                    direction = _restore_shape(direction, shp)
                    p.add_(direction, alpha=lr)
                else:
                    # vector / bias: l_infty base => sign step
                    p.add_(torch.sign(g_eff), alpha=-lr)

        return loss


# ----------------------------
# Adaptive diagonal methods (Table 1 parameterizations)
# ----------------------------

class _DiagPowerEMA(Optimizer):
    """
    Generic diagonal-preconditioned momentum method:
      m_t = beta1 m_{t-1} + (1-beta1) g
      v_t = beta2 v_{t-1} + (1-beta2) g^2
      update = m_hat / (v_hat^power + eps)
    with decoupled weight decay (AdamW style) if weight_decay != 0.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        power: float = 0.5,   # Adam corresponds to power=0.5 (sqrt(v))
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(lr=lr, betas=betas, power=power, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            power = group["power"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                # decoupled WD
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # bias correction (kept to match repo tuning style)
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                denom = v_hat.pow(power).add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


class AdamTable1(_DiagPowerEMA):
    """Adam (Table 1): D_t = (EMA[g^2])^{1/4} => denom uses v^{1/2} => power=0.5"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, power=0.5, eps=eps, weight_decay=weight_decay)


class AdamSANIA(_DiagPowerEMA):
    """Adam-SANIA (Table 1): D_t = (EMA[g^2])^{1/2} => denom uses v^{1} => power=1.0"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, power=1.0, eps=eps, weight_decay=weight_decay)


class MADGRADTable1(_DiagPowerEMA):
    """MADGRAD (Table 1 parameterization): D_t = (EMA[g^2])^{1/6} => denom uses v^{1/3} => power=1/3"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, power=(1.0 / 3.0), eps=eps, weight_decay=weight_decay)


class AdaGradTable1(Optimizer):
    """
    AdaGrad (Table 1): D_t = (sum g^2)^{1/4} => denom uses (sum g^2)^{1/2}.
    This is standard AdaGrad (diagonal accumulator + sqrt).
    """
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "sum_sq" not in state:
                    state["sum_sq"] = torch.zeros_like(p)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                sum_sq = state["sum_sq"]
                sum_sq.addcmul_(g, g, value=1.0)
                denom = sum_sq.sqrt().add_(eps)
                p.addcdiv_(g, denom, value=-lr)

        return loss


# ----------------------------
# Quasi-Newton / Kronecker (2D Shampoo/KFAC family)
# ----------------------------

class _KroneckerBase(Optimizer):
    """
    Shared machinery for KFAC/Shampoo/OneSided/KL-Shampoo.
    Applies to 2D parameters; other tensors fall back to AdamTable1 by default.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        eps=1e-8,
        update_freq: int = 1,
        beta: float = -1.0,  # beta<0 => sum; beta in [0,1) => EMA
        max_rows: int = 10000,
        fallback_adam: bool = True,
        adam_betas=(0.9, 0.999),
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, eps=eps, update_freq=update_freq,
            beta=beta, max_rows=max_rows, fallback_adam=fallback_adam, adam_betas=adam_betas
        )
        super().__init__(params, defaults)

        # Create a fallback diagonal optimizer for non-matrix params if requested
        self._fallback = None
        if fallback_adam:
            self._fallback = AdamTable1(params, lr=lr, betas=adam_betas, eps=eps, weight_decay=weight_decay)

    def _should_matrix(self, p: torch.Tensor, group) -> bool:
        return _use_matrix_rules(p, max_rows=group["max_rows"])

    def _fallback_step(self):
        if self._fallback is not None:
            self._fallback.step()

    # subclasses implement _matrix_update(p, Gm, state, group)


class ShampooTable1(_KroneckerBase):
    """
    Shampoo (Table 1): A_t = sum G G^T ; B_t = sum G^T G
    update: A^{-1/4} G B^{-1/4}
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        any_fallback = False

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta = group["beta"]
            update_freq = max(1, int(group["update_freq"]))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._should_matrix(p, group):
                    any_fallback = True
                    continue

                Gm, shp = _as_matrix(p.grad)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    m, n = Gm.shape
                    state["A"] = torch.zeros((m, m), device=Gm.device, dtype=Gm.dtype)
                    state["B"] = torch.zeros((n, n), device=Gm.device, dtype=Gm.dtype)
                    state["A_inv4"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype)
                    state["B_inv4"] = torch.eye(n, device=Gm.device, dtype=Gm.dtype)

                state["step"] += 1
                t = state["step"]

                # decoupled WD
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                A = state["A"]
                B = state["B"]

                outerA = Gm @ Gm.T
                outerB = Gm.T @ Gm

                if beta is not None and beta >= 0.0:
                    A.lerp_(outerA, 1.0 - beta)
                    B.lerp_(outerB, 1.0 - beta)
                else:
                    A.add_(outerA)
                    B.add_(outerB)

                if (t % update_freq) == 0:
                    state["A_inv4"] = _inv_root_psd(A, root=4.0, eps=eps)
                    state["B_inv4"] = _inv_root_psd(B, root=4.0, eps=eps)

                A_inv4 = state["A_inv4"]
                B_inv4 = state["B_inv4"]

                update = (A_inv4 @ Gm) @ B_inv4
                update = _restore_shape(update, shp)
                p.add_(update, alpha=-lr)

        if any_fallback:
            self._fallback_step()

        return loss


class OneSidedShampooTable1(_KroneckerBase):
    """
    One-sided Shampoo (Table 1): L = (sum G G^T)^{1/4}, R = I.
    Effective update: (L^T L)^{-1} G = (A^{1/2})^{-1} G = A^{-1/2} G
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        any_fallback = False

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta = group["beta"]
            update_freq = max(1, int(group["update_freq"]))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._should_matrix(p, group):
                    any_fallback = True
                    continue

                Gm, shp = _as_matrix(p.grad)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    m, _n = Gm.shape
                    state["A"] = torch.zeros((m, m), device=Gm.device, dtype=Gm.dtype)
                    state["A_inv2"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype)

                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                A = state["A"]
                outerA = Gm @ Gm.T
                if beta is not None and beta >= 0.0:
                    A.lerp_(outerA, 1.0 - beta)
                else:
                    A.add_(outerA)

                if (t % update_freq) == 0:
                    state["A_inv2"] = _inv_root_psd(A, root=2.0, eps=eps)  # A^{-1/2}

                update = state["A_inv2"] @ Gm
                update = _restore_shape(update, shp)
                p.add_(update, alpha=-lr)

        if any_fallback:
            self._fallback_step()

        return loss


class KFACTable1(ShampooTable1):
    """
    K-FAC (Table 1): same two-sided structure but with EMA expectation.
    Implemented by ShampooTable1 with beta in [0,1).
    """
    pass


class KLShampooTable1(_KroneckerBase):
    """
    KL-Shampoo (Table 1):
      A_t = EMA[ G (B^{-1}) G^T ]
      B_t = EMA[ G^T (A^{-1}) G ]

    We maintain A,B and cached A_inv, B_inv, and inv fourth roots for the update.
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        any_fallback = False

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta = group["beta"]
            if beta is None or beta < 0.0:
                # KL-Shampoo is defined with EMA; default to 0.9 if not provided
                beta = 0.9
            update_freq = max(1, int(group["update_freq"]))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._should_matrix(p, group):
                    any_fallback = True
                    continue

                Gm, shp = _as_matrix(p.grad)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    m, n = Gm.shape
                    state["A"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype) * eps
                    state["B"] = torch.eye(n, device=Gm.device, dtype=Gm.dtype) * eps
                    state["A_inv"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype)
                    state["B_inv"] = torch.eye(n, device=Gm.device, dtype=Gm.dtype)
                    state["A_inv4"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype)
                    state["B_inv4"] = torch.eye(n, device=Gm.device, dtype=Gm.dtype)

                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                A = state["A"]
                B = state["B"]
                A_inv = state["A_inv"]
                B_inv = state["B_inv"]

                # Update coupled statistics using previous inverses
                outerA = Gm @ B_inv @ Gm.T
                outerB = Gm.T @ A_inv @ Gm

                A.lerp_(outerA, 1.0 - beta)
                B.lerp_(outerB, 1.0 - beta)

                if (t % update_freq) == 0:
                    # refresh inverses and inv fourth roots
                    state["A_inv"] = _sym_matrix_power(A, power=-1.0, eps=eps)
                    state["B_inv"] = _sym_matrix_power(B, power=-1.0, eps=eps)
                    state["A_inv4"] = _inv_root_psd(A, root=4.0, eps=eps)
                    state["B_inv4"] = _inv_root_psd(B, root=4.0, eps=eps)

                update = (state["A_inv4"] @ Gm) @ state["B_inv4"]
                update = _restore_shape(update, shp)
                p.add_(update, alpha=-lr)

        if any_fallback:
            self._fallback_step()

        return loss


# ----------------------------
# Hybrid: SPlus (SOAP eigenbasis + sign LMO in rotated space)
# ----------------------------

class SPlusTable1(_KroneckerBase):
    """
    SPlus (Table 1): Q_L, Q_R eigenbases from Shampoo covariances; base norm l1 -> linf (sign).
    We implement 2D only:
      - maintain A = sum/EMA(GG^T), B = sum/EMA(G^T G)
      - compute eigenvectors QL, QR at update_freq
      - project: G' = QL^T G QR
      - sign LMO: S' = -sign(G')
      - unproject: S = QL S' QR^T
      - update: W <- W + lr * S   (S already includes descent sign)
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        any_fallback = False

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta = group["beta"]
            update_freq = max(1, int(group["update_freq"]))

            for p in group["params"]:
                if p.grad is None:
                    continue
                if not self._should_matrix(p, group):
                    any_fallback = True
                    continue

                Gm, shp = _as_matrix(p.grad)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    m, n = Gm.shape
                    state["A"] = torch.zeros((m, m), device=Gm.device, dtype=Gm.dtype)
                    state["B"] = torch.zeros((n, n), device=Gm.device, dtype=Gm.dtype)
                    state["QL"] = torch.eye(m, device=Gm.device, dtype=Gm.dtype)
                    state["QR"] = torch.eye(n, device=Gm.device, dtype=Gm.dtype)

                state["step"] += 1
                t = state["step"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                A = state["A"]
                B = state["B"]
                outerA = Gm @ Gm.T
                outerB = Gm.T @ Gm
                if beta is not None and beta >= 0.0:
                    A.lerp_(outerA, 1.0 - beta)
                    B.lerp_(outerB, 1.0 - beta)
                else:
                    A.add_(outerA)
                    B.add_(outerB)

                if (t % update_freq) == 0:
                    # eigenbases
                    wL, QL = torch.linalg.eigh(0.5 * (A + A.T))
                    wR, QR = torch.linalg.eigh(0.5 * (B + B.T))
                    state["QL"] = QL
                    state["QR"] = QR

                QL = state["QL"]
                QR = state["QR"]

                Gp = (QL.T @ Gm) @ QR
                Sp = lmo.sign_lmo(Gp)  # already descent
                S = (QL @ Sp) @ (QR.T)
                S = _restore_shape(S, shp)
                p.add_(S, alpha=lr)

        if any_fallback:
            self._fallback_step()

        return loss
