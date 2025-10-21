# %%
import afbf
from afbf import tbfield
import numpy as np
import matplotlib.pylab as plt

def gen_hfbf(N, H, return_field = False):
    # Define the field.
    field = tbfield('fbf')

    # Change the parameter of the Hurst function.
    field.hurst.ChangeParameters(np.array([[H]]))
    field.NormalizeModel()

    # Simulate the field.
    z_sdata = field.Simulate()
    z = z_sdata.values.reshape(z_sdata.M)
    if return_field:
        return z, field
    return z

# T largeur
# D direction
def gen_efbf(N,  H, t=1.0, d=1.0, return_field = False):
    T = t*np.pi / 6  # Interval bound for selected frequencies.
    D = d*np.pi / 3  # Direction.
    # Define the field.
    # Mode of simulation for step values (alt, 'unif', 'unifmax', or 'unifrange').

    # Define the field to be simulated and coordinates where to simulate.
    field = tbfield('efbf')
    coord = afbf.coordinates(N)
    field.hurst.ChangeParameters(fparam=np.array([[H]]))
    field.topo.ChangeParameters(fparam=np.array([0, 1]), finter=np.array([-T, T]))
    # Translate the topothesy function to be at the right orientation.
    field.topo.ApplyTransforms(translate=-D)
    simu = field.Simulate(coord)
    # Simulate the field.
    z = simu.values.reshape(simu.M)
    if return_field:
        return z, field
    return z
    
def gen_afbf(N,  return_field = False):
    # Define the field.
    # Mode of simulation for step values (alt, 'unif', 'unifmax', or 'unifrange').
    simstep = 'unifmin'
    # Mode of simulation for step interval bounds (alt, 'nonunif').
    simbounds = 'unif'

    # Define the field to be simulated and coordinates where to simulate.
    field = tbfield('afbf-smooth')
    coord = afbf.coordinates(N)
    field.hurst.SetStepSampleMode(mode_cst=simstep, mode_int=simbounds)
    
    #np.random.seed(seed)
    field.hurst.ChangeParameters()
    field.topo.ChangeParameters()

    #np.random.seed(seed)
    field.EvaluateTurningBandParameters()
    simu = field.Simulate(coord)
    # Simulate the field.
    z = simu.values.reshape(simu.M)
    if return_field:
        return z, field
    return z


################################################
##################DKL###########################
################################################


def A_func(tau, h):
    return tau/(2*h)

def B_func(tau,h):
    return  tau / ((h)**2)

def compute_normalization(tau, h):
    integrand = A_func(tau, h)
    return np.sum(integrand)/len(integrand)

def get_tau_h(X):
    t = np.linspace(0,2*np.pi,2000, dtype=float)
    X.hurst.Evaluate(t)
    X.topo.Evaluate(t)    
    
    h     = X.hurst.values[0]
    tau   = X.topo.values[0]

    return tau, h

def d_KL(X_1, X_2): ##### Assuming normalized field
    tau_1, h_1 = get_tau_h(X_1)
    tau_2, h_2 = get_tau_h(X_2)

    return d_KL_with_tau_h(tau_1, h_1, tau_2, h_2)

def d_KL_with_tau_h(tau_1, h_1, tau_2, h_2):
    A = A_func(tau_1,h_1)        
    log_rat = log_ratios(tau_1, tau_2)

    d_tau_integrand = np.multiply(A,log_rat)
    d_tau_integral = np.sum(d_tau_integrand)/len(d_tau_integrand)


    B = B_func(tau_1, h_1)       
    diff_h = h_2 - h_1

    d_h_integrand = np.multiply(B, diff_h) 
    d_h_integral = np.sum(d_h_integrand)/len(d_h_integrand)
    #print(f'd_tau_integrand is {d_tau_integrand}')
    #print(f'd_h_integrand is {d_h_integrand}')

    return d_tau_integral +d_h_integral, d_tau_integral, d_h_integral

def d_KL_for_unnormalized(X_1, X_2):
    tau_1, h_1 = get_tau_h(X_1)
    tau_2, h_2 = get_tau_h(X_2)
    
    Z_1 = compute_normalization(tau_1, h_1)
    Z_2 = compute_normalization(tau_2, h_2)

    d, _, _ = d_KL_with_tau_h(tau_1, h_1, tau_2, h_2)
    return (1/Z_1)*(d) + np.log(Z_2/Z_1)
    #return Z_1*(d + np.log(Z_1/Z_2))

def d_KL_sym_for_unnormalized(X_1, X_2):
    d12 = d_KL_for_unnormalized(X_1,X_2)
    d21 = d_KL_for_unnormalized(X_2,X_1)

    return d12 + d21

def d_KL_sym(X_1,X_2):
    d12, d_tau_12, d_hurst_12 = d_KL(X_1,X_2)
    d21, d_tau_21, d_hurst_21 = d_KL(X_2,X_1)
    return d12 + d21, d_tau_12 + d_tau_21, d_hurst_12 + d_hurst_21


def A_sym(tau_1,h_1, tau_2, h_2):
    f = lambda tau,h : tau / (2*h)
    A_1 = f(tau_1,h_1)
    A_2 = f(tau_2, h_2)

    return A_1 - A_2

def log_ratios(tau_1, tau_2):
    return np.log(tau_1/tau_2)

def B_sym(tau_1,h_1, tau_2, h_2):
    f = lambda tau,h : tau / ((h)**2)
    B_1 = f(tau_1, h_1)
    B_2 = f(tau_2, h_2)

    return B_1 - B_2

       
def display_params(z, Z, path, filename, title = ""):
    theta_list = np.linspace(-np.pi,np.pi,2000, dtype=float)
    Z.hurst.Evaluate(theta_list)

    h = Z.hurst.values[0]
    
    #fig, axs = plt.subplots(1,3,subplot_kw={'projection':'polar'}, squeeze=False)
    #fig, axs = plt.subplots(1,3, squeeze=False)
    fig = plt.figure()
    ax0 = fig.add_subplot(1,3,1)
    #axs[0,0].remove()
    #ax = fig.add_subplot(1,3,1, projection='3d')
    ax0.imshow(z)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.title.set_text(title)


    #axs[0,1].remove()
    #ax = fig.add_subplot(1,3,2,projection='polar')
    ax1 = fig.add_subplot(1,3,2, projection='polar')
    xT=ax1.get_xticks()
    xL=['$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    ax1.set_frame_on(False)
    ax1.plot(theta_list, h)
    ax1.set_rmax(1.1)
    ax1.set_rticks([0.25,0.5,0.75,1.0])
    ax1.set_rlabel_position(-35.5)  # Move radial labels away from plotted line
    ax1.set_xticks(xT, xL)
    ax1.grid(True)


    Z.topo.Evaluate(theta_list)    
    tau = Z.topo.values[0]

    #axs[0,2].remove()
    #ax = fig.add_subplot(1,3,3, projection='polar')
    ax2 = fig.add_subplot(1,3,3, projection='polar')
    xT=ax2.get_xticks()
    xL=['$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    ax2.set_frame_on(False)
    ax2.plot(theta_list, tau)
    ax2.set_rlim(0)
    ax2.set_rscale('symlog')
    #ax2.set_rticks([1,4,16,64,128], ['$2^0$', '$2^2$','$2^4$','$2^6$','$2^7$'])
    ax2.set_rticks([1,4,16,64,128], ['','','','',''])
    ax2.set_rlabel_position(-35.5)  # Move radial labels away from plotted line
    ax2.set_xticks(xT, xL)
    ax2.grid(True)
    #ax.set_title(f'$\tau_{idx}$', va='bottom')

    fig.set_tight_layout(True)
    fig.savefig(path + filename + '.pdf', transparent=True, dpi=600, format='pdf')

def display_params_indiv(z, Z, path, filename, title = ""):
    theta_list = np.linspace(-np.pi,np.pi,2000, dtype=float)
    Z.hurst.Evaluate(theta_list)

    h = Z.hurst.values[0]
    
    #fig, axs = plt.subplots(1,3,subplot_kw={'projection':'polar'}, squeeze=False)
    #fig, axs = plt.subplots(1,3, squeeze=False)
    fig0 = plt.figure()
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax0 = fig0.gca()
    #axs[0,0].remove()
    #ax = fig.add_subplot(1,3,1, projection='3d')
    ax0.imshow(z)
    ax0.set_xticks([])
    ax0.set_yticks([])
    #ax0.title.set_text(title)


    #axs[0,1].remove()
    #ax = fig.add_subplot(1,3,2,projection='polar')
    ax1 = fig1.add_subplot(1,1,1, projection='polar')
    xT=ax1.get_xticks()
    xL=['$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    ax1.set_frame_on(False)
    ax1.plot(theta_list, h)
    ax1.set_rmax(1.1)
    ax1.set_rticks([0.25,0.5,0.75,1.0], ['', '', '', ''])
    #ax1.set_ylabel(['a', 'b', 'c', 'd'])
    #ax1.set_rticks([1.0])
    ax1.set_rlabel_position(-35.5)  # Move radial labels away from plotted line
    ax1.set_xticks(xT, xL)
    ax1.grid(True)


    Z.topo.Evaluate(theta_list)    
    tau = Z.topo.values[0]

    #axs[0,2].remove()
    #ax = fig.add_subplot(1,3,3, projection='polar')
    ax2 = fig2.add_subplot(1,1,1, projection='polar')
    xT=ax2.get_xticks()
    xL=['$0$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{2}$',r'$\frac{3\pi}{4}$',r'$\pi$',r'$\frac{5\pi}{4}$',r'$\frac{3\pi}{2}$',r'$\frac{7\pi}{4}$']
    ax2.set_frame_on(False)
    ax2.plot(theta_list, tau)
    ax2.set_rlim(0)
    ax2.set_rscale('symlog')
    #ax2.set_rticks([1,4,16,64,128], ['$2^0$', '$2^2$','$2^4$','$2^6$','$2^7$'])
    ax2.set_rticks([1,4,16,64,128], ['', '','','',''])
    ax2.set_rlabel_position(-35.5)  # Move radial labels away from plotted line
    ax2.set_xticks(xT, xL)
    ax2.grid(True)
    #ax.set_title(f'$\tau_{idx}$', va='bottom')

    fig0.set_tight_layout(True)
    fig0.savefig(path + filename + 'mixture' + '.pdf', transparent=True, dpi=600, format='pdf')
    fig1.set_tight_layout(True)
    fig1.savefig(path + filename + 'hurst' + '.pdf', transparent=True, dpi=600, format='pdf')
    fig2.set_tight_layout(True)
    fig2.savefig(path + filename + 'tau' + '.pdf', transparent=True, dpi=600, format='pdf')

def RotateModel(field, rotation = 0.0, img_size = 512):
    
    new_field = afbf.tbfield(fname = field.fname, topo = clone_perfunction(field.topo), hurst = clone_perfunction(field.hurst))

    new_field.topo.ApplyTransforms(translate = rotation)
    new_field.hurst.ApplyTransforms(translate = rotation)

    img_grid = afbf.coordinates()
    img_grid.DefineUniformGrid(img_size)

    arr = new_field.Simulate(coord=img_grid)
    z = arr.values.reshape(arr.M)

    return z, new_field

def clone_perfunction(perf):
    new_perf = afbf.perfunction()
    new_perf.ftype   = perf.ftype
    new_perf.fparam  = perf.fparam
    new_perf.fname   = perf.fname
    new_perf.finter = perf.finter
    new_perf.steptrans = perf.steptrans
    new_perf.trans = perf.trans
    new_perf.translate = perf.translate
    new_perf.rescale = perf.rescale

    return new_perf


# %%
if __name__ == "__main__":
    H = 0.7
    t = 1
    d = 2
    hfbf_f, hfbf_p = gen_hfbf(256,0.5, True)
    efbf_f, efbf_p = gen_efbf(256, H, t, d, True)
    afbf_f, afbf_p = gen_afbf(256, True)
    fields_n = ["hfbf", "efbf", "afbf"]
    fields_f = [hfbf_f, efbf_f, afbf_f]
    fields_p = [hfbf_p, efbf_p, afbf_p]
#%% 
    fig, axs = plt.subplots(1,3, figsize = [3,9])
    for idx in range(len(fields_f)):
        axs[idx].imshow(fields_f[idx])
        axs[idx].axis(False)
    
        display_params_indiv(fields_f[idx], fields_p[idx], ".", "fname", title = fields_n[idx])

# %%
    rotation_list = [i*np.pi/10 for i in range(5)]
    fig, axs = plt.subplots(1,len(rotation_list), figsize = [3,9])
    rot_f_list = []
    rot_p_list = []
    for idx, rot in enumerate(rotation_list):
        rot_f, rot_p = RotateModel(afbf_p, rot)
        rot_f_list.append(rot_f)
        rot_p_list.append(rot_p)
        axs[idx].imshow(rot_f)
        axs[idx].axis(False)
    
        display_params_indiv(rot_f, rot_p, ".", rot_p.fname, title = rot_p.fname)
# %%
    d_list = []
    for p_1 in rot_p_list:
        d_list.append([])
        for p_2 in rot_p_list:
            d  = d_KL_sym_for_unnormalized(p_1, p_2)
            print(d)
            d_list[-1].append(d)
    plt.figure()
    for idx in range(len(rot_p_list)):
        plt.plot(d_list[idx])

