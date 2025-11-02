#%% 
import deepinv as dinv
import torch
import wandb
import h5py

from lir3a.HetTextures import HetTexture
from lir3a.CPTV import LinRegCP, make_param_groups
from lir3a.utils import update_h5py

def RegularizeAndLearn(het_text, epochs=10, K_steps=50, R_restarts=10, learning_rate=1e-1,  lambda_init = 1e2, device = "cpu", h5_filepath ="", learn_weights = False):

    model = LinRegCP(J_scales = het_text.J_scales, B_bands = 6, rho = 0.5, lambd = lambda_init, K_steps = K_steps, R_restarts = R_restarts, learn_weights = learn_weights, device = device)

    model_h5_savefile = h5py.File(h5_filepath, 'a')


    ####### Training initialization #######
    project = 'texture_segmentation'

    loss_fn = dinv.loss.SupLoss(metric=dinv.metric.MSE())

    optimizer_name = "ADAM"
    
    if learn_weights :
        param_groups = make_param_groups(model, 1.0, learning_rate, learning_rate*0.05)
        optimizer = torch.optim.Adam(param_groups, weight_decay=1e-8)
    else : 
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

    config_dict = {
        'lambda_init' : lambda_init,  
        'learning_rate' : learning_rate,
        'K_steps' : K_steps,
        'R_restarts' : R_restarts,
        'learn_weights' : learn_weights,
        'lambda_init' : lambda_init,
        'epochs' : epochs,
        'optimizer' : optimizer_name
    }
    for key in config_dict:
        print(key)
        update_h5py(model_h5_savefile, key, config_dict[key])
    wandb.init(name = "name", project=project, group = "group", config = config_dict)

    ####### Training loop #######
    loss_list = []
    lambda_list = []
    weight_list = []
    crit_list = []

    # Inserting batch dimensions
    truths = het_text.ground_truth[None, ...]
    observations = het_text.het_text_coefs.L[None, ...]
    
    for epoch in range(epochs):
        
        print(f'epoch {epoch}/{epochs}')

        x, x_tilde, u, crit = model(observations, ret_crit = True)

        loss = loss_fn(x, truths).sum()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        epoch_loss = loss.item()
        loss_list.append(loss.cpu().detach().numpy())
        lambda_list.append(model.elambda.cpu().detach().numpy())
        if learn_weights:
            weight_list.append(model.fin_diff_phys.log_weight.clone().cpu().detach().numpy())
        crit_list.append(crit_list)
                
        print(f'loss is {loss.item()},  lambda is {lambda_list[-1]}')
        wandb.log({"epoch": epoch, "train_loss" : loss.item()})
        
        for name, param in model.named_parameters():
            wandb.log({f"params/{name}": param.clone().cpu().detach().numpy()})  # Logging mean value

        torch.cuda.empty_cache()
        wandb.log({"epoch_loss": epoch_loss})
##Test
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    wandb.finish()
    #torch.save(model.state_dict(), savemodels_dir + model_name + '.pt')
    update_h5py(model_h5_savefile, 'train/learned_weights', weight_list)
    update_h5py(model_h5_savefile, 'train/loss_list', loss_list)
    update_h5py(model_h5_savefile, 'train/lambda_list', lambda_list)
    update_h5py(model_h5_savefile, 'train/final_x', x.cpu().detach().numpy())
    model_h5_savefile.close()

    return x

# %%
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from lir3a.GenTextures import gen_hfbf
    from lir3a.GenMasks import gen_brownian_square_mask
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    J_scales = 6

    H_1 = 0.75
    H_2 = 0.65
    hfbf_1, hfbf_1_p = gen_hfbf(512,H_1, True)
    hfbf_2, hfbf_2_p = gen_hfbf(512,H_2, True)
    hfbf_1 = normalize_field(hfbf_1)
    hfbf_2 = normalize_field(hfbf_2)
    brownian_mask = gen_brownian_square_mask(width = 100, scale = 25, img_size = 512)

    crop = True
    het_text = HetTexture(hfbf_1, hfbf_2, brownian_mask, J_scales, device, crop = crop)
    het_text.compute_difficulties(1.0, hfbf_1_p, hfbf_2_p)
    plt.figure()
    plt.imshow(het_text.het_texture)

    # %%
    het_text.to_h5py("test_het_text.h5")
    het_text.eval_prediction(het_text.het_text_coefs.F)
    #%% 
    
    epochs = 10
    K_steps = 50
    R_restarts = 10
    learning_rate = 1e-1
    lambda_init = -1.0

    x_pred = RegularizeAndLearn(het_text, epochs, K_steps, R_restarts, learning_rate,  lambda_init, device, "test_het_text.h5", learn_weights = False)
    het_text.eval_prediction(x_pred[0,...])

# %%
    het_text.to_h5py("test_het_text_learn_weights.h5")

    #%% 
    
    epochs = 10
    K_steps = 50
    R_restarts = 10
    learning_rate = 1e-1
    lambda_init = -1.0

    x_pred = RegularizeAndLearn(het_text, epochs, K_steps, R_restarts, learning_rate,  lambda_init, device, "test_het_text_learn_weights.h5", learn_weights = True)
    het_text.eval_prediction(x_pred[0,...])

# %%
