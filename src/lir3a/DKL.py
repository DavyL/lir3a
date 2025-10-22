import numpy as np

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
