# %% 

import deepinv as dinv
import torch


class LinRegPhysics(dinv.physics.LinearPhysics):
    """ SHAPES : (BATCH x) BAND x SPACE x SCALES/COEFS 
    x corresponds to target linear regression coefficients stored as x_{batch, band, x, y} = (v, h)
    y corresponds to observation (wavelet coeffs or equiv. their least square fit prediction) y_{batch, band, x, y} = (c_1, ..., c_J)
"""
    def __init__(
        self,
        J_scales,
        B_bands,
        device,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self.J_scales = J_scales
        self.B_bands = B_bands
        self.A_star_A = None
        self.A_star_A_inv = None
        self.init_matrices(device=device)

    def A(self,x):
        new_shape = list(x.shape[:-1])
        new_shape.append(self.J_scales)
        y = torch.empty(new_shape, dtype = x.dtype, device = x.device)
        for j in range(self.J_scales):
            y[..., j] = x[..., 0] + j * x[..., 1] 
        return y


    def A_adjoint(self, y):
        new_shape = list(y.shape[:-1])
        new_shape.append(2)
        x_ret = torch.zeros(size=new_shape, dtype = y.dtype, device = y.device)
        if y.shape[-1] != self.J_scales:
            print(f'In LinRegPhysics:A_adjoint: number of scales of y ({y.shape[-1]}) and J_scales ({self.J_scales}) do not coincide')

        for j in range(self.J_scales):
            x_ret[..., 0] += y[..., j]
            x_ret[..., 1] += j*y[..., j]

        return x_ret
    
    def init_matrices(self, device, dtype=torch.float32):
        A_star_A = torch.empty(size=[2,2], device=device, dtype=dtype)
        
        R_0 = 0
        R_1 = 0
        R_2 = 0
        for j in range(self.J_scales):
            R_0 += 1 
            R_1 += j 
            R_2 += j**2
        det = R_2 * R_0 - R_1 **2
        A_star_A[0,0] = R_0
        A_star_A[0,1] = R_1
        A_star_A[1,0] = R_1
        A_star_A[1,1] = R_2

        
        A_star_A_inv = torch.empty_like(A_star_A)
        A_star_A_inv[0,0] = R_2 / det
        A_star_A_inv[1,0] = -R_1 / det
        A_star_A_inv[0,1] = -R_1 / det
        A_star_A_inv[1,1] = R_0 / det

        self.A_star_A = A_star_A
        self.A_star_A_inv = A_star_A_inv

        return A_star_A, A_star_A_inv

    def A_dagger(self, y):

        A_star_y = self.A_adjoint(y)

        for b in range(self.B_bands):
            x_dagger = torch.einsum('...ij, ...bkli -> ...bklj', self.A_star_A_inv, A_star_y)
        return x_dagger
        
if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    batch = 3
    B = 7
    J = 14
    phys = LinRegPhysics(J_scales = J, B_bands=B, device = device)
    x_shape = (B,260,270,2)
    x = torch.ones(x_shape, device = device)
    x[...,0] = 12.0
    x_obs = x 
    y = phys.A(x_obs)
    eps = torch.randn_like(y, device =device)
    y = y + eps
    adj_y = phys.A_adjoint(y)
    dagg_y = phys.A_dagger(y)
    print(phys.adjointness_test(x_obs))
    print(dagg_y)
# %%
