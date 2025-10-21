# %%

import torch 
import deepinv as dinv
from LinRegPhysics import LinRegPhys
from LinRegDataFid import BatchedMultiScaleDF
from FinDiffPhysics import NablaPhys
from deepinv.optim import L12Prior

class LinRegCP(torch.nn.Module):
    def __init__(self, J_scales, B_bands, rho, lambd, K_steps, R_restarts, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lin_reg_phys = LinRegPhys(J_scales = J, B_bands=B, device = device)

        self.data_fidelity = BatchedMultiScaleDF(lin_reg_phys)
        self.prior = L12Prior(l2_axis=(1,-2,-1))

        self.fin_diff_phys = NablaPhys()

        self.sigma_init   = 0.5   /   torch.sqrt(torch.tensor(2.0))
        self.tau_init     = 0.5   /   torch.sqrt(torch.tensor(2.0))
        self.rho    = rho
        self.lambd = torch.nn.Parameter(torch.tensor(lambd, device=device), requires_grad = True)

        self.K_steps   = K_steps
        self.R_restarts  = R_restarts


    def forward(self, y, ret_crit = False):    

        self.elambda    = torch.exp(self.lambd)
        x,     x_tilde,     u = self.set_init(y)

        crit_list = []
        with torch.no_grad():
            for r in range(self.R_restarts):
                print(f'{r}/{self.R_restarts}')
                x, x_tilde, u = self.R_step(x, y, x_tilde, u, crit_list)

        self.elambda    = torch.exp(self.lambd)

        x, x_tilde, u = self.R_step(x, y, x_tilde, u, crit_list)

        if ret_crit:
            return x, x_tilde, u, crit_list
        return x, x_tilde, u

    def set_init(self, y):
        x_init  = self.lin_reg_phys.A_dagger(y)
        x_tilde = torch.zeros_like(x_init)
        u       = self.fin_diff_phys.A(x_init) 

        return x_init, x_tilde, u

    def R_step(self, x, y, x_tilde, u, crit_list):
        sigma   = self.sigma_init
        tau     = self.tau_init

        for k in range(K_steps):
            u_next = self.prior.prox_conjugate(u + sigma * self.fin_diff_phys.A(x_tilde), gamma = sigma, lamb = self.elambda)
            x_next = self.data_fidelity.prox(x - tau*self.fin_diff_phys.A_adjoint(u_next), y, gamma = tau)

            chi     = 1.0   /   (torch.sqrt(1+2*self.rho*tau))
            tau     = tau * chi
            sigma   = sigma / chi

            x_tilde_next = (1+chi)*x_next - chi*x

            crit_val = self.criterion(y, x_next)
            crit_list.append(crit_val.cpu().detach().numpy()) 

            x       = x_next
            x_tilde = x_tilde_next
            u       = u_next
                
        return x, x_tilde, u

    def criterion(self, y, x):
        df  = self.data_fidelity.d(x, y)
        reg = self.elambda*self.prior.fn(self.fin_diff_phys.A(x))

        return df+reg
    

#%% 
if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    batch = 2
    B = 3
    J = 8
    x_shape = (batch, B, 500, 500, 2)
    x_true = torch.ones(x_shape, device = device)  
    x_true[...,0] = 12.0
    x_true[..., 150:300, : ,0] = 10.0
    x_true[...,1] = 1.2
    x_true[..., 300:450, : ,1] = 1.0
    x_obs = x_true 
    lin_reg_phys = LinRegPhys(J_scales = J, B_bands=B, device = device)

    y = lin_reg_phys.A(x_obs)
    eps = torch.randn_like(y)
    y = y + eps

    K_steps = 50
    R_restarts = 10

    rho     = 0.5
    lambd = 10.0

#%%
    model = LinRegCP(J_scales = J, B_bands = B, rho = rho, lambd = lambd, K_steps = K_steps, R_restarts = R_restarts, device = device)
    x, x_tilde, u, crit = model(y, ret_crit = True)
# %%
    import matplotlib.pyplot as plt
    plt.figure()
    x_np = x.cpu().detach().numpy()
    x_true_np = x_true.cpu().detach().numpy()
    fig, axs = plt.subplots(B, 2, figsize = [4, 2*B])
    for b in range(B):
        for v in range(2):
            axs[b,v].imshow(x_np[0,b, ...,v])

    plt.figure()
    plt.plot(crit)

    plt.figure()
    plt.hist(x_np[0,..., 0].flatten(), bins = 200)
    plt.hist(x_true_np[0,..., 0].flatten(), bins = 100)
    plt.figure()
    plt.hist(x_np[0,..., 1].flatten(), bins = 200)
    plt.hist(x_true_np[0,..., 1].flatten(), bins = 100)
# %%
