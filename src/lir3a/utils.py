import numpy as np


def update_h5py(loc, key, new_val):
    if key in loc.keys():
        del loc[key]
    loc[key] = new_val

def hor_forward(x):
    """ Horizontal forward finite differences (with Neumann boundary conditions) """
    hor = np.zeros_like(x)
    hor[:,:-1] = x[:,1:] - x[:,:-1]
    return hor

def ver_forward(x):
    """ Vertical forward finite differences (with Neumann boundary conditions) """
    ver = np.zeros_like(x)
    ver[:-1,:] = x[1:,:] - x[:-1,:]
    return ver

def dir_op(x):
    h = 0.5*hor_forward(x)
    v = 0.5*ver_forward(x)
    return np.stack((h,v), 2)