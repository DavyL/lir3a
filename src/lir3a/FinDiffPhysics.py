# %%

import torch 
import deepinv as dinv

def nabla(I):####A vérifier
    b, c, h, w, f = I.shape
    G = torch.zeros((b, c, h, w, f, 2), device=I.device).type(I.dtype)
    G[:, :, :-1, :, :, 0] = G[:, :, :-1, :, :, 0] - I[:    , :     , :-1]
    G[:, :, :-1, :, :, 0] = G[:, :, :-1, :, :, 0] + I[:    , :     , 1:]
    G[:, :, :, :-1, :, 1] = G[:, :, :, :-1, :, 1] - I[...  , :-1   , :]
    G[:, :, :, :-1, :, 1] = G[:, :, :, :-1, :, 1] + I[...  , 1:    , :]
    return G

def nablaT(G):
    b, c, h, w, f = G.shape[:-1]
    I = torch.zeros((b, c, h, w, f), device=G.device).type(
        G.dtype
    )  
    I[:, :, :-1,:]    = I[:, :, :-1]  - G[:, :, :-1, :, :, 0]
    I[:, :, 1:,:]     = I[:, :, 1:]   + G[:, :, :-1, :, :, 0]
    I[..., :-1,:]     = I[:, :, :, :-1,:]   - G[:, :, :, :-1, :, 1]
    I[..., 1:,:]      = I[:, :, :, 1:,:]    + G[:, :, :, :-1, :, 1]
    return I

class NablaPhys(dinv.physics.LinearPhysics):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def A(self, x, **kwargs):
        return nabla(x)

    def A_adjoint(self, y, **kwargs):
        return nablaT(y)

if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    batch = 1
    B = 7
    J = 14
    x_shape = (batch, B,260,270,2)
    x = torch.ones(x_shape, device = device)  
    x = x + torch.randn_like(x, device =device)
    x[...,0] = 12.0

    fin_diff_physics = NablaPhys(device = device)
    fin_diff = fin_diff_physics.A(x)
    #print(f'{fin_diff[...,:,:,:].shape}') 
    #print(f'{fin_diff_physics.A_adjoint(fin_diff)}') 
    print(f'{fin_diff_physics.adjointness_test(x)}')
# %%
