# %%
import torch 
import deepinv as dinv

from dtcwt.utils import compute_wavelet_coefs ## Can be found at https://github.com/DavyL/undecimated_dtcwt
from lir3a.LinRegPhysics import LinRegPhys


class WavCoefs():
    """ Class to store wavelet coefs C associated to a field X. 
    C is a B x N_x x N_y x J complex torch tensor of dtcwt coefficients (B = 6 orientations, J scales)
    L is the log-modulus wavelet field associated to C
    F is the (V,H) feature field associated to pointwise linear regression of F over scales
    F_pw is a B x 1 x 1 x 2 tensor corresponding correspond to the spatial average value of F (log first, regression and mean last)
    F_glob is a B x 1 x 1 x 2 tensor corresponding correspond to the value of F computed over spatially averaged L (mean first, log and regression last)
    """
    def __init__(self, X, J_scales, device, crop = False):
        self.X = X
        self.image_size = X.shape
        self.J_scales   = J_scales
        self.lin_reg_phys = LinRegPhys(J_scales = J_scales, B_bands=6, device = device)
        if crop :
            self.crop_width = -2**(J_scales - 1)
            self.C      = torch.tensor(self.compute_dtcwt(X), device = device)[:,:self.crop_width, :self.crop_width, ...]
        else :
            self.crop_width = -1
            self.C      = torch.tensor(self.compute_dtcwt(X), device = device)
            
        self.L      = torch.log2(torch.abs(self.C)).to(dtype = torch.float32)
        self.F      = self.compute_features_from_coefs(self.L)
        self.F_glob = self.compute_global_features_from_avg_coefs(self.C)

        self.F_pw   = torch.mean(self.F, dim = (1,2), keepdim = True) ## Used as ground truth for learning

    def compute_dtcwt(self, X):
        coefs = compute_wavelet_coefs(X, nlevels = self.J_scales)
        return coefs

    def compute_features_from_coefs(self, L):
        return self.lin_reg_phys.A_dagger(L)

    def compute_global_features_from_avg_coefs(self, C):
        avg = torch.log2(torch.mean(torch.abs(C).to(dtype = torch.float32), dim = (1,2), keepdim = True))
        return self.lin_reg_phys.A_dagger(avg)
# %%
if __name__ == "__main__":
    from lir3a.GenTextures import gen_hfbf

    H = 0.7
    hfbf_f, hfbf_p = gen_hfbf(512,H, True)

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    J_scales = 6
    coefs = WavCoefs(hfbf_f, J_scales, device)
# %%
    print(f'avg first features : {coefs.F_glob}')
    print(f'log first features : {coefs.F_pw}')

    expanded_F_glob = coefs.F_glob.expand(*coefs.F.shape)
    expanded_F_pw = coefs.F_pw.expand(*coefs.F.shape)

    print(f'glob error : {torch.mean(torch.square(expanded_F_glob - coefs.F))}')
    print(f'pw error : {torch.mean(torch.square(expanded_F_pw - coefs.F))}')

    pred_F_glob = coefs.lin_reg_phys.A(expanded_F_glob)
    pred_F_pw = coefs.lin_reg_phys.A(expanded_F_pw)

    print(f'glob pred error : {torch.mean(torch.square(pred_F_glob - coefs.L))}')
    print(f'pw pred error : {torch.mean(torch.square(pred_F_pw - coefs.L))}')
# %%
