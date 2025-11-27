# %%

import torch 
import deepinv as dinv
from lir3a.LinRegPhysics import LinRegPhys
from lir3a.LinRegDataFid import BatchedMultiScaleDF
from lir3a.FinDiffPhysics import NablaPhys, WeightedNablaPhys
from deepinv.optim import L12Prior



class UnrolledCP(torch.nn.Module):
    def __init__(self, A_phys, data_fid, C_channels = 3, F_features = 2, rho = 0.5, lambd = -1.0, K_steps = 1000, R_restarts = 0, learn_weights = False, device = "cpu", *args, **kwargs):
        super().__init__()
        
        self.lin_reg_phys = A_phys
        self.data_fidelity = data_fid
        self.prior = L12Prior(l2_axis=(1,-2,-1))

        if learn_weights :
            self.fin_diff_phys = WeightedNablaPhys(C= C_channels, F = F_features)
        else:
            self.fin_diff_phys = NablaPhys()

        self.sigma_init   = 0.5   /   torch.sqrt(C_channels * torch.tensor(2.0))
        self.tau_init     = 0.5   /   torch.sqrt(C_channels * torch.tensor(2.0))
        self.rho    = rho
        self.lambd = torch.nn.Parameter(torch.tensor(lambd, device=device), requires_grad = True)

        self.K_steps   = K_steps
        self.R_restarts  = R_restarts

        self.to(device)

    def forward(self, y, ret_crit = False):    

        self.elambda    = torch.exp(self.lambd)
        x,     x_tilde,     u = self.set_init(y)

        crit_list = []
        with torch.no_grad():
            for r in range(self.R_restarts):
                #print(f'{r}/{self.R_restarts}')
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

        for k in range(self.K_steps):
            u_next = self.prior.prox_conjugate(u + sigma * self.fin_diff_phys.A(x_tilde), gamma = sigma, lamb = self.elambda)
            x_next = self.data_fidelity.prox(x - tau*self.fin_diff_phys.A_adjoint(u_next), y, gamma = tau, physics = self.lin_reg_phys)

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
    

class LinRegCP(UnrolledCP):
    def __init__(self, J_scales, B_bands = 6,device = "cpu", *args, **kwargs):
        lin_reg_phys = LinRegPhys(J_scales = J_scales, B_bands=B_bands,  device = device)
        data_fidelity = BatchedMultiScaleDF(lin_reg_phys)
        super().__init__(A_phys=lin_reg_phys, data_fid = data_fidelity, C_channels = B_bands, *args, **kwargs)

class DenoisingCP(UnrolledCP):
    def __init__(self, *args, **kwargs):
        lin_reg_phys = dinv.physics.Denoising()
        data_fidelity = dinv.optim.L2()
        super().__init__(A_phys=lin_reg_phys, data_fid = data_fidelity, *args, **kwargs)

def make_param_groups(model, lr_main: float, lr_lambda: float, lr_weights: float):
    """
    Build parameter groups for an optimizer, assigning different learning rates to:
        - general model parameters      -> lr_main
        - lambda (model.lambd)          -> lr_lambda
        - WeightedNablaPhys parameters  -> lr_weights

    Args:
        model: LinRegCP instance (or compatible)
        lr_main:    learning rate for all other parameters
        lr_lambda:  learning rate for model.lambd
        lr_weights: learning rate for model.fin_diff_phys (if WeightedNablaPhys)

    Returns:
        List of parameter-group dicts ready to pass to an optimizer.
    """

    param_groups = []

    # 1️⃣ Base parameters (everything except lambda and fin_diff_phys)
    base_params = [
        p for n, p in model.named_parameters()
        if not (n.startswith("lambd") or n.startswith("fin_diff_phys"))
    ]
    if base_params:
        param_groups.append({"params": base_params, "lr": lr_main})

    # 2️⃣ Lambda parameter (if learnable)
    if hasattr(model, "lambd") and isinstance(model.lambd, torch.nn.Parameter):
        param_groups.append({"params": [model.lambd], "lr": lr_lambda})

    # 3️⃣ WeightedNablaPhys parameters (if present and learnable)
    if hasattr(model, "fin_diff_phys"):
        fin_diff_phys = model.fin_diff_phys
        # Only include if it's actually a WeightedNablaPhys (has parameters)
        if any(True for _ in fin_diff_phys.parameters()):
            param_groups.append({
                "params": fin_diff_phys.parameters(),
                "lr": lr_weights
            })

    return param_groups
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
    lin_reg_phys = LinRegPhys(J_scales = J, B_bands=B, learn_weights = False, device = device)

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
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    C_channels = 3
    file = "rof_seg_2_11/"

    # %%
    #brownian_mask = gen_brownian_square_mask(width = 125, scale = 25, img_size = 512)
    x_true = torch.ones([512,512]).to(device)
    sigma_noise = 1.0
    noise_physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma = sigma_noise))
    #x_true = torch.from_numpy(mask).float()  # -> [N_x, N_y]
    x_true = x_true.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # -> [1, 1, N_x, N_y, 1]
    x_true = x_true.repeat(1, 1, 1, 1, C_channels)           # -> [1, C_channels, N_x, N_y, 1]y       = noise_physics(x_true)
    x_true += 1.0
    x_true[...,0]   *= 0.25
    x_true[...,1]   *= 0.5
    x_true *= 0.5
    y = noise_physics(x_true)
    x_true_np = x_true.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
#    plt.figure()
#    plt.imshow(x_true[0,0])
#    plt.axis('off')
#
#    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # supprime tout padding interne
#    plt.savefig(file + 'x_true_field.png', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
#
#    plt.figure()
#    plt.imshow(y[0,0])
#    plt.axis('off')
#
#    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # supprime tout padding interne
#    plt.savefig(file + 'y_field.png', bbox_inches='tight', pad_inches=0, dpi=300 , transparent=True)



    # %%
    model = DenoisingCP(C_channels= 1, F_features = C_channels, rho = 0.5, lambd = 5.0, K_steps = 10000, R_restarts = 0, device = device)

    model.sigma_init   = 0.5   /   (torch.sqrt(torch.tensor(2.0)))
    model.tau_init     = 0.5   /   (torch.sqrt(torch.tensor(2.0)))
    with torch.no_grad():
        x, x_tilde, u, crit = model(y, ret_crit = True)
# %%
