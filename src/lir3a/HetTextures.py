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

from sklearn.cluster import KMeans

def get_bins(mask_true, mask_pred):
    true_positive = np.sum(np.multiply(mask_true, mask_pred))
    true_negative = np.sum(np.multiply(1.0-mask_true, 1.0 - mask_pred))
    false_positive = np.sum(np.multiply(1.0-mask_true, mask_pred))
    false_negative = np.sum(np.multiply(mask_true, 1.0-mask_pred))
    return true_positive, true_negative, false_positive, false_negative

class HetTexture():
    def __init__(self, X1, X2, mask, J_scales, device, crop=False, filter_length_mask = 51, sigma_mask = 1.0):
        self.img_size   = mask.shape
        self.J_scales   = J_scales
        self.device     = device
        self.X1     = X1
        self.X2     = X2
        self.mask   = mask
        self.het_texture = apply_smoothed_mask(X1, X2, mask, filter_length_mask, sigma_mask)
        self.het_text_coefs = WavCoefs(self.het_texture, J_scales, device, crop = crop)
        
        self.lin_reg_phys   = LinRegPhys(J_scales = J_scales, B_bands=6, device = device)

        self.X1_features    = WavCoefs(X1, J_scales, device, crop = crop).F_pw
        self.X2_features    = WavCoefs(X2, J_scales, device, crop = crop).F_pw
        
        if crop :
            crop_width  = self.het_text_coefs.crop_width
            self.X1     = self.X1[:crop_width, :crop_width]
            self.X2     = self.X2[:crop_width, :crop_width]
            self.mask   = self.mask[:crop_width, :crop_width]
            self.het_texture   = self.het_texture[:crop_width, :crop_width]

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

    def F_score(self, mask_pred):
        mask_true = self.mask
        true_positive, true_negative, false_positive, false_negative = get_bins(mask_true, mask_pred)
        if true_positive + true_negative < false_negative + false_positive:
            mask_pred = 1.0 - mask_pred
            true_positive, true_negative, false_positive, false_negative = get_bins(mask_true, mask_pred)

        sensitivity = true_positive / np.sum(mask_true)
        specificity = true_positive / np.sum(mask_pred)

        f_score = 2* (sensitivity * specificity)/(sensitivity + specificity)

        Delta = (true_positive + false_positive)*(false_negative + true_negative)*(true_positive + false_negative)*(false_positive + true_negative)
        MCC_1 = (true_positive*true_negative - false_positive*false_negative)/np.sqrt(Delta)

        mask_pred = 1.0 - mask_pred
        true_positive, true_negative, false_positive, false_negative = get_bins(mask_true, mask_pred)
        Delta = (true_positive + false_positive)*(false_negative + true_negative)*(true_positive + false_negative)*(false_positive + true_negative)
        MCC_2 = (true_positive*true_negative - false_positive*false_negative)/np.sqrt(Delta)

        return f_score, max(MCC_1, MCC_2)


    def eval_prediction(self, pred, K =2):
        """
        pred: torch.Tensor or np.ndarray of shape (C, H, W, F)
        K:    number of clusters
        returns:
            labels_hw: (H, W) integer labels in [0..K-1]
            centers_c_f: (K, C, F) cluster centers reshaped
            kmeans: fitted KMeans object
        """
        x = pred.detach().cpu().numpy()

        assert x.ndim == 4, f"Expected pred of shape (C,H,W,F), got {x.shape}"

        C, H, W, F = x.shape

        # 2) reshape to (N_samples, N_features) = (H*W, C*F)
        X = np.ascontiguousarray(x.transpose(1, 2, 0, 3).reshape(H * W, C * F))  # (H,W,C,F)->(H*W, C*F)


        kmeans = KMeans(n_clusters=K, n_init="auto")

        kmeans.fit(X)
        labels = kmeans.labels_.reshape(H, W)

        centers_cf = kmeans.cluster_centers_.reshape(K, C, F)
        
        self.labels = labels
        self.centers_cf = centers_cf

        f_score, mcc = self.F_score(self.labels)

        return mcc



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

#%% 
    het_text = HetTexture(hfbf_1, hfbf_2, brownian_mask, J_scales, device)
    mc = het_text.eval_prediction(het_text.het_text_coefs.F)
    print(mc)
    het_text.compute_difficulties(1.0, hfbf_1_p, hfbf_2_p)
    plt.figure()
    plt.imshow(het_text.het_texture)
    plt.figure()
    plt.imshow(het_text.labels)

    # %%
    het_text.to_h5py("test_het_text.h5")

# %%
    from lir3a.CPTV import LinRegCP
    K_steps = 50
    R_restarts = 10

    rho     = 0.5
    lambd = 10.0

    model = LinRegCP(J_scales = J_scales, B_bands = 6, rho = rho, lambd = lambd, K_steps = K_steps, R_restarts = R_restarts, device = device, learn_weights = False)
    x, x_tilde, u, crit = model(het_text.het_text_coefs.L[None,...], ret_crit = True)
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
