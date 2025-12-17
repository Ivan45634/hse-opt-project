# src/optimizers/lmo.py
import torch
from muon import zeropower_via_newtonschulz5 as _muon_zeropower

@torch.no_grad()
def frobenius_lmo(G: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Base LMO for Frobenius norm ball:
      argmin_{||S||_F <= 1} <G, S> = - G / ||G||_F
    We return the *direction* (normalized gradient) with sign already applied.
    """
    n = torch.norm(G)
    return -G / (n + eps)


@torch.no_grad()
def sign_lmo(G: torch.Tensor) -> torch.Tensor:
    """
    Base LMO for l1 -> linf / vector linf-style direction in practice:
    returns negative sign direction.
    """
    return -torch.sign(G)


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 10,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Newton–Schulz iteration to approximate the polar factor of a matrix.
    This is the same core routine used by Muon/TAIA-style spectral LMO.

    Input: 2D tensor G (m x n)
    Output: approximate polar(G) ≈ U V^T (spectral LMO direction)
    """
    assert G.ndim == 2, "zeropower_via_newtonschulz5 expects a 2D matrix"
    m, n = G.shape

    # Scale to improve numerical stability
    # Use Frobenius norm scaling
    frob = torch.norm(G)
    if frob < eps:
        return torch.zeros_like(G)

    X = G / (frob + eps)

    # If m < n, work on transpose and transpose back for stability
    transposed = False
    if m < n:
        X = X.T
        transposed = True

    # Newton–Schulz iterations for inverse sqrt
    # This variant approximates polar factor without explicit SVD.
    I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    Y = X
    Z = I

    for _ in range(steps):
        # T = 0.5 * (3I - ZY)
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    # Y is approximately orthonormal/polar factor
    P = Y

    if transposed:
        P = P.T

    return P


@torch.no_grad()
def spectral_lmo(
    G: torch.Tensor,
    ns_steps: int = 10,
    eps: float = 1e-7,
    scale_rectangular: bool = True,
) -> torch.Tensor:
    """
    Spectral-norm LMO direction via Muon's Newton–Schulz polar approximation.
    Works for rectangular matrices (Muon implementation handles it).
    Returns descent direction (negative).
    """
    assert G.ndim == 2
    P = _muon_zeropower(G, steps=ns_steps, eps=eps)

    if scale_rectangular:
        m, n = G.shape
        P = P * (max(1.0, float(m) / float(n)) ** 0.5)

    return -P


@torch.no_grad()
def d_precond_transform(G: torch.Tensor, D: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Applies D^{-1} ⊙ G (elementwise). D must be broadcastable to G.
    """
    return G / (D + eps)


@torch.no_grad()
def d_precond_lmo(G: torch.Tensor, D: torch.Tensor, base_lmo_fn, eps: float = 1e-12) -> torch.Tensor:
    """
    LMO under D-preconditioned norm:
        lmo_{D,base}(G) = D^{-1} ⊙ lmo_base(D^{-1} ⊙ G)
    """
    Gt = d_precond_transform(G, D, eps=eps)
    S = base_lmo_fn(Gt)
    return d_precond_transform(S, D, eps=eps)


@torch.no_grad()
def lr_precond_lmo(G: torch.Tensor, L_inv: torch.Tensor, R_inv: torch.Tensor, base_lmo_fn) -> torch.Tensor:
    """
    LMO under (L,R)-preconditioned norm:
        lmo_{L,R,base}(G) = L^{-1} * lmo_base(L^{-T} G R^{-T}) * R^{-1}

    Here L_inv = L^{-1}, R_inv = R^{-1}.
    """
    Gt = (L_inv.T @ G) @ (R_inv.T)
    S = base_lmo_fn(Gt)
    return (L_inv @ S) @ (R_inv)
