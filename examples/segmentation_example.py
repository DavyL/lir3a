
#%% 
if __name__ == "__main__":
    import deepinv as dinv
    import torch
    import wandb
    import h5py
    
    import numpy as np
    import matplotlib.pyplot as plt

    from lir3a.HetTextures import HetTexture
    from lir3a.CPTV import LinRegCP, make_param_groups
    from lir3a.LearnWeights import RegularizeAndLearn
    from lir3a.utils import update_h5py
    from lir3a.GenTextures import gen_afbf, normalize_field
    from lir3a.GenMasks import gen_brownian_square_mask

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    J_scales = 6


    afbf_1, afbf_1_p = gen_afbf(512, True)
    afbf_2, afbf_2_p = gen_afbf(512, True)
    afbf_1 = normalize_field(afbf_1)
    afbf_2 = normalize_field(afbf_2)
    brownian_mask = gen_brownian_square_mask(width = 100, scale = 25, img_size = 512)

    crop = True
    het_text = HetTexture(afbf_1, afbf_2, brownian_mask, J_scales, device, crop = crop)
    het_text.compute_difficulties(1.0, afbf_1_p, afbf_2_p)
    plt.figure()
    plt.imshow(het_text.het_texture)

    het_text.to_h5py("test_het_text_learn_weights.h5")

    #%% 
    
    epochs = 100
    K_steps = 50
    R_restarts = 10
    learning_rate = 1e-1
    lambda_init = -1.0

    x_pred = RegularizeAndLearn(het_text, epochs, K_steps, R_restarts, learning_rate,  lambda_init, device, "test_het_text_learn_weights.h5", learn_weights = True)
    mcc = het_text.eval_prediction(x_pred[0,...])
    print(f'MCC : {mcc}')

    plt.figure()
    plt.imshow(het_text.labels)
# %%
    x_pred = RegularizeAndLearn(het_text, epochs, K_steps, R_restarts, learning_rate,  lambda_init, device, "test_het_text_fixed_weights.h5", learn_weights = False)
    mcc = het_text.eval_prediction(x_pred[0,...])
    print(f'MCC : {mcc}')
    plt.figure()
    plt.imshow(het_text.labels)

# %%
