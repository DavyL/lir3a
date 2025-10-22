# %%
import torch 
import deepinv as dinv
from LinRegPhysics import LinRegPhys
from LinRegDataFid import BatchedMultiScaleDF
from FinDiffPhysics import NablaPhys
from deepinv.optim import L12Prior
import matplotlib.pylab as plt

def CP(K_steps, x, x_tilde, u, y, F, G, L, lambd, sigma, tau, rho):
    """
    Strongly convex Chambolle-Pock algorithm to minimize F(x, y) + lambd*G(Lx).
    K_steps : number of steps
    x, x_tilde : primal and inertial variables
    u : dual variable
    F, G : Data fidelity and prior
    L : Linear operator (dinv.physics)
    sigma, tau : initial stepsizes
    rho : strong convexity constant
    """
    crit = []
    for k in range(K_steps):
        print(f'{k}/{K_steps}')
        u_next = G.prox_conjugate(u + sigma * L.A(x_tilde), gamma = sigma, lamb = lambd)
        x_next = F.prox(x - tau*L.A_adjoint(u_next), y, gamma = tau)

        chi     = 1.0   /   (torch.sqrt(1+2*rho*tau))
        tau     = tau * chi
        sigma   = sigma / chi

        x_tilde_next = (1+chi)*x_next - chi*x

        crit_val = 0.5*F.d(x_next, y) + lambd*G.fn(L.A(x_next))
        crit.append(crit_val.cpu().detach().numpy()) 

        x = x_next
        x_tilde = x_tilde_next
        u = u_next
            
    return x, x_tilde, u, crit


#%% 
if __name__ == "__main__":
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    batch = 2
    B = 3
    J = 8
    x_shape = (batch, B,500,500,2)
    x_true = torch.ones(x_shape, device = device)  
    x_true[...,0] = 12.0
    x_true[..., 150:300, : ,0] = 10.0
    x_true[...,1] = 1.2
    x_true[..., 300:450, : ,1] = 1.0
    x_obs = x 
    lin_reg_phys = LinRegPhys(J_scales = J, B_bands=B, device = device)

    y = lin_reg_phys.A(x_obs)
    eps = torch.randn_like(y)
    y = y + eps

    fin_diff_physics = NablaPhys()

    df = BatchedMultiScaleDF(lin_reg_phys)

    K_steps = 2000
    sigma   = 0.5   /   torch.sqrt(torch.tensor(2.0))
    tau     = 0.5   /   torch.sqrt(torch.tensor(2.0))
    rho     = 0.5
    lambd = 100.0
    l12_prior = L12Prior(l2_axis=(1,-2,-1))
    
    x_init = lin_reg_phys.A_dagger(y)
    x_tilde = torch.zeros_like(x_init)
    u = fin_diff_physics.A(x_init) 


    x, x_tilde, u, crit = CP(K_steps, x_init, x_tilde, u, y, df, l12_prior, fin_diff_physics, lambd, sigma, tau, rho)
# %%
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
    plt.hist(x_true_np[0,..., 0].flatten(), bins = 10)
    plt.figure()
    plt.hist(x_np[0,..., 1].flatten(), bins = 200)
    plt.hist(x_true_np[0,..., 1].flatten(), bins = 10)
# %%
