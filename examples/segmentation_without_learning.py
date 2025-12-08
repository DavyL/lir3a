#%%
import torch 
import deepinv as dinv

from lir3a.AnalyzeTextures import WavCoefs
from lir3a.CPTV import LinRegCP
from lir3a.LinRegPhysics import LinRegPhys
from lir3a.LinRegDataFid import BatchedMultiScaleDF
from lir3a.FinDiffPhysics import NablaPhys
from lir3a.HetTextures import HetTexture, compute_K_means
from lir3a.GenTextures import gen_afbf, normalize_field
from lir3a.GenMasks import gen_brownian_square_mask
from deepinv.optim import L12Prior
import matplotlib.pylab as plt

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

J_scales = 6


# %%
# Code below generates a heterogeneous texture
afbf_1, afbf_1_p = gen_afbf(512, True)
afbf_2, afbf_2_p = gen_afbf(512, True)
afbf_1 = normalize_field(afbf_1)
afbf_2 = normalize_field(afbf_2)
brownian_mask = gen_brownian_square_mask(width = 100, scale = 25, img_size = 512)

crop = True
het_text = HetTexture(afbf_1, afbf_2, brownian_mask, J_scales, device, crop = crop)
het_text.compute_difficulties(1.0, afbf_1_p, afbf_2_p)

#%%

## Loading the heterogeneous texture array and computing wavelet coefficients object
het_text_array = het_text.het_texture
plt.figure()
plt.imshow(het_text_array)

coefs = WavCoefs(het_text_array, J_scales, device)

h_v_no_reg = coefs.F
log_coefs = coefs.L


CP_alg = LinRegCP(J_scales = J_scales, B_bands=6, K_steps = 100, lambd =1.0, device = device)


y = log_coefs[None,...] #Extending with an empty batch dimension

x, x_tilde, u, crit = CP_alg(y, ret_crit = True)
# %%
# Plotting texture, criterion through iterations, histograms of h,v estimated values
plt.figure()
x_np = x.cpu().detach().numpy()
fig, axs = plt.subplots(6, 2, figsize = [4, 2*6])
for b in range(6):
    for v in range(2):
        axs[b,v].imshow(x_np[0,b, ...,v])

plt.figure()
plt.plot(crit)

plt.figure()
plt.hist(x_np[0,..., 0].flatten(), bins = 200)
plt.figure()
plt.hist(x_np[0,..., 1].flatten(), bins = 200)


# %%
segmentation, _ = compute_K_means(x[0])

plt.figure()
plt.imshow(segmentation)
# %%
