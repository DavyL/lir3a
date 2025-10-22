#%%
import deepinv as dinv
import torch
import h5py
import numpy as np

from lir3a.GenMasks import apply_smoothed_mask, apply_mask
from lir3a.AnalyzeTextures import WavCoefs
from lir3a.LinRegPhysics import LinRegPhys
from lir3a.utils import update_h5py, dir_op
from lir3a.DKL import d_KL_sym_for_unnormalized

class HetTexture():
    def __init__(self, X1, X2, mask, J_scales, device):
        self.img_size   = mask.shape
        self.J_scales   = J_scales
        self.device     = device
        self.X1     = X1
        self.X2     = X2
        self.mask   = mask
        self.het_texture = apply_smoothed_mask(X1, X2, mask, filter_length = 51)

        self.lin_reg_phys   = LinRegPhys(J_scales = J_scales, B_bands=6, device = device)

        self.X1_features    = WavCoefs(X1, J_scales, device).F_pw
        self.X2_features    = WavCoefs(X2, J_scales, device).F_pw
        self.het_text_coefs = WavCoefs(self.het_texture, J_scales, device)
        self.compute_ground_truth()
        self.dkl = None
        self.beta = None
        self.segmentation_difficulty = None

    def compute_difficulties(self, beta, X1_param, X2_param):
        self.compute_mask_difficulty()
        self.compute_texture_divergence(X1_param=X1_param, X2_param=X2_param)
        self.compute_segmentation_difficulty(beta)
        return
    
    def compute_ground_truth(self):
        print(self.X1_features.shape, self.X2_features.shape, torch.tensor(self.mask).shape)
        mask    = torch.as_tensor( self.mask, dtype   = self.X1_features.dtype, device = self.X1_features.device).view(1, *self.mask.shape, 1)          

        self.ground_truth = apply_mask(self.X1_features, self.X2_features, torch.tensor(mask))
        return self.ground_truth

    def compute_mask_difficulty(self):
        self.mask_difficulty = np.sum(np.abs(dir_op(self.mask)))

        return self.mask_difficulty

    def compute_texture_divergence(self, X1_param, X2_param):
        self.dkl = d_KL_sym_for_unnormalized(X1_param, X2_param)
        return self.dkl

    def compute_segmentation_difficulty(self, beta):
        self.beta = beta
        self.segmentation_difficulty = np.exp(-beta * self.dkl) * self.mask_difficulty
        return self.segmentation_difficulty

    def to_h5py(self, path, mode = 'a'):
        file = h5py.File(path, mode)
        update_h5py(file, "img_size", self.img_size)
        update_h5py(file, "J_scales", self.J_scales)
        update_h5py(file, "X1",     self.X1)
        update_h5py(file, "X2",     self.X2)
        update_h5py(file, "mask",   self.mask)
        update_h5py(file, "beta",   self.beta)
        update_h5py(file, "mask_difficulty",            self.mask_difficulty)
        update_h5py(file, "segmentation_difficulty",    self.segmentation_difficulty)
        update_h5py(file, "het_texture",    self.het_texture)
        update_h5py(file, "X1_features",    self.X1_features.cpu().detach().numpy())
        update_h5py(file, "X2_features",    self.X2_features.cpu().detach().numpy())
        update_h5py(file, "het_text_log_coefs", self.het_text_coefs.L.cpu().detach().numpy())
        update_h5py(file, "ground_truth", self.ground_truth.cpu().detach().numpy())

        del file
# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def normalize_field(X):
        return (X - np.mean(X))#/np.var(X)

    from lir3a.GenTextures import gen_hfbf
    from lir3a.GenMasks import gen_brownian_square_mask
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    J_scales = 6

    H_1 = 0.75
    H_2 = 0.7
    hfbf_1, hfbf_1_p = gen_hfbf(512,H_1, True)
    hfbf_2, hfbf_2_p = gen_hfbf(512,H_2, True)
    hfbf_1 = normalize_field(hfbf_1)
    hfbf_2 = normalize_field(hfbf_2)
    brownian_mask = gen_brownian_square_mask(width = 100, scale = 25, img_size = 512)

    het_texture = HetTexture(hfbf_1, hfbf_2, brownian_mask, J_scales, device)
    het_texture.compute_difficulties(1.0, hfbf_1_p, hfbf_2_p)
    plt.figure()
    plt.imshow(het_texture.het_texture)

    # %%
    het_texture.to_h5py("test_het_text.h5")

# %%
    from lir3a.CPTV import LinRegCP
    K_steps = 50
    R_restarts = 10

    rho     = 0.5
    lambd = 10.0

    model = LinRegCP(J_scales = J_scales, B_bands = 6, rho = rho, lambd = lambd, K_steps = K_steps, R_restarts = R_restarts, device = device)
    x, x_tilde, u, crit = model(het_texture.het_text_coefs.L[None,...], ret_crit = True)
# %%
    plt.figure()
    x_np = x.cpu().detach().numpy()
    #x_true_np = x_true.cpu().detach().numpy()
    fig, axs = plt.subplots(6, 2, figsize = [4, 2*6])
    for b in range(6):
        for v in range(2):
            axs[b,v].imshow(x_np[0,b, ...,v])

    plt.figure()
    plt.plot(crit)
# %%
