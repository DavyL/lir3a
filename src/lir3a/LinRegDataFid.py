# %%

import torch
import deepinv as dinv
import LinRegPhysics as LR



class BatchedMultiScaleDF(dinv.optim.DataFidelity):
    def __init__(self, lin_reg_phys, **kwargs):
        super(dinv.optim.DataFidelity, self).__init__(**kwargs)

        self.lin_phys = lin_reg_phys
        self.A_star_A = lin_reg_phys.A_star_A
        self.J_scales = lin_reg_phys.J_scales
        self.B_bands = lin_reg_phys.B_bands

    def d(self,u,y,*args, **kwargs):
        df = 0.5*torch.sum(torch.square(y - self.lin_phys.A(u)), axis = [1,2,3,4])
        return df

    def prox(self, x, y, *args, gamma=1.0,  **kwargs):
        mat = torch.linalg.inv(gamma*self.lin_phys.A_star_A + torch.diag(torch.ones(2, device = self.A_star_A.device)))
        C = x + gamma*self.lin_phys.A_adjoint(y)
        ret = torch.einsum('ij, zbkli -> zbklj', mat, C)

        return ret
    
    def prox_conjugate(self, x, y, physics, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""

        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma (\lambda \datafidname)^*}(x)`,
            computed in :math:`x`.
        """
        return x - gamma * self.prox(
            x / gamma, y, physics, *args, gamma=lamb / gamma, **kwargs
        )

if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    batch = 1
    B = 7
    J = 14
    phys = LR.LinRegPhysics(J_scales = J, B_bands=B, device = device)
    x_shape = (batch, B,260,270,2)
    x = torch.ones(x_shape, device = device)
    x[...,0] = 12.0
    x_obs = x 
    y = phys.A(x_obs)
    eps = torch.randn_like(y, device =device)
    y = y + eps
    adj_y = phys.A_adjoint(y)
    dagg_y = phys.A_dagger(y)
    print(f'dagg y is {dagg_y}')

    df = BatchedMultiScaleDF(phys)
    print(f'd is {df.d(x, y)}')
    print(f'prox is {df.prox(x, y, physics = None, gamma = 1.0)}')


# %%
