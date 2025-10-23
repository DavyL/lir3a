# %%

import torch 
import deepinv as dinv
import torch.nn as nn
import math
# ================================================================
# 1. Finite difference operator  and its adjoint
# ================================================================

def nabla(I: torch.Tensor) -> torch.Tensor:
    """
    Compute forward finite differences along the spatial dimensions (h, w).

    Args:
        I: Tensor of shape (B, C, H, W, F)
           B = batch, C = channels, H = height, W = width, F = extra dimension (e.g., bands, features)

    Returns:
        G: Tensor of shape (B, C, H, W, F, 2)
           G[..., 0] = forward difference along H (vertical)
           G[..., 1] = forward difference along W (horizontal)
    """
    B, C, H, W, F = I.shape

    # Forward difference along height (h)
    dh = torch.zeros((B, C, H, W, F), device=I.device, dtype=I.dtype)
    dh[:, :, :-1] = I[:, :, 1:] - I[:, :, :-1]

    # Forward difference along width (w)
    dw = torch.zeros((B, C, H, W, F), device=I.device, dtype=I.dtype)
    dw[:, :, :, :-1] = I[:, :, :, 1:] - I[:, :, :, :-1]

    # Stack along a new dimension for the two gradient directions
    G = torch.stack((dh, dw), dim=-1)
    return G


def nablaT(G: torch.Tensor) -> torch.Tensor:
    """
    Compute the adjoint (divergence) of the forward finite difference operator.

    Args:
        G: Tensor of shape (B, C, H, W, F, 2)
           G[..., 0] = gradient along H (dh)
           G[..., 1] = gradient along W (dw)

    Returns:
        I: Tensor of shape (B, C, H, W, F)
           The divergence, i.e., the adjoint of nabla.
    """
    B, C, H, W, F = G.shape[:-1]
    dh, dw = G[..., 0], G[..., 1]

    I = torch.zeros((B, C, H, W, F), device=G.device, dtype=G.dtype)

    # Adjoint of forward difference along height
    # Subtract on pixel i, add on pixel i+1
    I[:, :, :-1] -= dh[:, :, :-1]
    I[:, :, 1:]  += dh[:, :, :-1]

    # Adjoint of forward difference along width
    I[:, :, :, :-1] -= dw[:, :, :, :-1]
    I[:, :, :, 1:]  += dw[:, :, :, :-1]

    return I

#def nabla(I):
#    b, c, h, w, f = I.shape
#    G = torch.zeros((b, c, h, w, f, 2), device=I.device).type(I.dtype)
#    G[:, :, :-1, :, :, 0] = G[:, :, :-1, :, :, 0] - I[:    , :     , :-1]
#    G[:, :, :-1, :, :, 0] = G[:, :, :-1, :, :, 0] + I[:    , :     , 1:]
#    G[:, :, :, :-1, :, 1] = G[:, :, :, :-1, :, 1] - I[...  , :-1   , :]
#    G[:, :, :, :-1, :, 1] = G[:, :, :, :-1, :, 1] + I[...  , 1:    , :]
#    return G
#
#def nablaT(G):
#    b, c, h, w, f = G.shape[:-1]
#    I = torch.zeros((b, c, h, w, f), device=G.device).type(
#        G.dtype
#    )  
#    I[:, :, :-1,:]    = I[:, :, :-1]  - G[:, :, :-1, :, :, 0]
#    I[:, :, 1:,:]     = I[:, :, 1:]   + G[:, :, :-1, :, :, 0]
#    I[..., :-1,:]     = I[:, :, :, :-1,:]   - G[:, :, :, :-1, :, 1]
#    I[..., 1:,:]      = I[:, :, :, 1:,:]    + G[:, :, :, :-1, :, 1]
#    return I

class NablaPhys(dinv.physics.LinearPhysics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def A(self, x, **kwargs):
        return nabla(x)

    def A_adjoint(self, y, **kwargs):
        return nablaT(y)
# ================================================================
# 2) Weighted operator with C×F weights and positive constraint
#    via exponential reparametrization.
# ================================================================

class WeightedNablaPhys(NablaPhys):
    """
    Weighted finite differences with learnable, positive weights over (C, F).
    Effective operator:
        A_w(x) = W ⊙ nabla(x)
        A_w^T(y) = nabla^T(W ⊙ y)

    where W > 0 and has shape:
        - (C, F, 1) if per_direction=False
        - (C, F, 2) if per_direction=True   (separate weights for h and w)

    We store free parameters `log_weight` in R and apply `exp` at use time:
        W = exp(log_weight)  (strictly positive, differentiable)
    """

    def __init__(self,
                 C: int,
                 F: int,
                 init_weight: float = 1.0,
                 per_direction: bool = False,
                 device=None,
                 dtype=None):
        """
        Args:
            C: number of channels
            F: size of the last dimension (bands/features)
            init_weight: positive scalar used to initialize exp-parameterization.
                         (If <=0, we use a small epsilon instead.)
            per_direction: if True, learn separate weights for h and w directions.
            device, dtype: optional device/dtype for parameter initialization.
        """
        super().__init__()
        self.per_direction = per_direction

        # Ensure strictly positive initial weight for the exp parameterization.
        eps = 1e-6
        w0 = max(init_weight, eps)
        logw0 = math.log(w0)

        if per_direction:
            # (C, F, 2): weight per (channel, feature, direction)
            shape = (C, F, 2)
        else:
            # (C, F, 1): single weight per (channel, feature), broadcast to 2 directions
            shape = (C, F, 1)

        # Unconstrained parameter in log-domain
        self.log_weight = nn.Parameter(torch.full(shape, logw0, device=device, dtype=dtype))

    def _effective_weight(self, ref: torch.Tensor) -> torch.Tensor:
        """
        Returns the positive weight tensor W ready to broadcast against nabla(x),
        with final shape broadcastable to (B, C, H, W, F, 2).

        Stored shape:
            (C, F, 1) or (C, F, 2)

        Returned view:
            (1, C, 1, 1, F, 2)  after exp + reshape
        """
        # exp ensures positivity; gradients flow through exp as usual
        Wcf2 = torch.exp(self.log_weight).to(device=ref.device, dtype=ref.dtype)   # (C,F,1 or 2)

        # If no per-direction, expand the last dim to 2 so it broadcasts to both directions
        if not self.per_direction and Wcf2.shape[-1] == 1:
            Wcf2 = Wcf2.expand(-1, -1, 2)  # (C,F,2) shared across directions

        # Reshape for broadcasting over (B,H,W)
        # (C,F,2) -> (1, C, 1, 1, F, 2)
        return Wcf2.view(1, Wcf2.shape[0], 1, 1, Wcf2.shape[1], 2)

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward operator:
            x:  (B, C, H, W, F)
            A(x): (B, C, H, W, F, 2)
        """
        G = nabla(x)                      # (B, C, H, W, F, 2)
        W = self._effective_weight(x)     # (1, C, 1, 1, F, 2), broadcastable
        return G * W

    def A_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Adjoint operator:
            y:  (B, C, H, W, F, 2)
            A^T(y): (B, C, H, W, F)
        """
        W = self._effective_weight(y)     # same broadcasting as in A
        return nablaT(y * W)

if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    batch = 1
    B = 7
    J = 14
    x_shape = (batch, B,260,270,2)
    x = torch.ones(x_shape, device = device)  
    x = x + torch.randn_like(x, device =device)
    x[...,0] = 12.0

    fin_diff_physics = WeightedNablaPhys(C=7, F=2, device = device)
    fin_diff = fin_diff_physics.A(x)
    #print(f'{fin_diff[...,:,:,:].shape}') 
    #print(f'{fin_diff_physics.A_adjoint(fin_diff)}') 
    print(f'{fin_diff_physics.adjointness_test(x)}')
# %%
