#%%
import scipy.signal as sg

import numpy as np

def apply_mask(z_1,z_2,mask):
    het_texture = z_1 + mask * (z_2 - z_1)
    return het_texture  


def gauss1d(sigma, filter_length=11):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter without normalization

    rng = range(-int(filter_length/2),int(filter_length/2)+1)
    gauss_filter = [np.exp((-x**2) / (2*sigma**2)) for x in rng]

    return np.array(gauss_filter).reshape(1,filter_length)

def smooth_mask(mask, sigma, filter_length = 11):
    _filter1d = gauss1d(sigma=sigma, filter_length=filter_length)
    _filter2d = _filter1d.T @ _filter1d
    _filter2d /= np.sum(np.abs(_filter2d))

    smoothed_mask = sg.convolve2d(mask, _filter2d, 'same')

    return smoothed_mask

def apply_smoothed_mask(z_1, z_2, mask, filter_length, sigma = 1.0):
    smoothed_mask = smooth_mask(mask=mask, sigma= sigma, filter_length=filter_length)

    smooth_mixture = apply_mask(z_1, z_2, smoothed_mask)

    return smooth_mixture


def gen_brownian_process(support, seed = 0):
    rng = np.random.default_rng(seed)
    dWt = rng.normal(loc = 0.0, scale = 1/np.sqrt(support), size = support)
    B = np.cumsum(dWt)
    return B

def gen_brownian_bridge(support):
    B1 = gen_brownian_process(support)
    B2 = gen_brownian_process(support)

    B = np.zeros_like(B1)
    for t in range(support):
        B[t] = (1-t/(support-1))*B1[t] + (t/(support-1))*B2[-t]
    return B


def fill_mask(mask):
    filled_mask = np.zeros_like(mask)
    for x in range(mask.shape[0]):## Fill horizontal
        keep_fill = True
        for y in range(mask.shape[1]):
            if(mask[x,y] != 0):
                keep_fill = False
            elif(keep_fill == True):
                filled_mask[x,y] = 1.0

    for y in range(mask.shape[1]):
        keep_fill = True
        for x in range(mask.shape[0]):
            if(mask[x,y] != 0):
                keep_fill = False
            elif(keep_fill == True):
                filled_mask[x,y] = 1.0

    for x in range(mask.shape[0]):## Fill horizontal
        keep_fill = True
        for y in range(mask.shape[1]):
            if(mask[x,-y] != 0):
                keep_fill = False
            elif(keep_fill == True):
                filled_mask[x,-y] = 1.0

    for y in range(mask.shape[1]):
        keep_fill = True
        for x in range(mask.shape[0]):
            if(mask[-x,y] != 0):
                keep_fill = False
            elif(keep_fill == True):
                filled_mask[-x,y] = 1.0
    
    return filled_mask

def gen_brownian_square_contour(corners, mask, scale):
    #corners = [midpoint + (64,64),midpoint + (-64,64),midpoint + (-64,-64),midpoint + (64,-64)]
    support = corners[1][1] - corners[2][1]

    B1 = gen_brownian_bridge(support)
    B2 = gen_brownian_bridge(support)
    B3 = gen_brownian_bridge(support)
    B4 = gen_brownian_bridge(support)
    start_slice = corners[2][1]
    for t in range(1,support):
        stop_slice = corners[2][1]+int(scale*B1[t])
        mask[corners[2][0]+t, min(start_slice,stop_slice):max(start_slice,stop_slice)+1] = [1]*(max(start_slice,stop_slice)+1 - min(start_slice,stop_slice))
        start_slice = stop_slice

    start_slice = corners[1][1]
    for t in range(1,support):
        stop_slice = corners[1][1]+int(scale*B2[t])
        mask[corners[1][0]+t, min(start_slice,stop_slice):max(start_slice,stop_slice)+1] = [1]*(max(start_slice,stop_slice)+1 - min(start_slice,stop_slice))
        start_slice = stop_slice

    start_slice = corners[2][0]
    for t in range(1,support):
        stop_slice = corners[2][0]+int(scale*B3[t])
        mask[min(start_slice,stop_slice):max(start_slice,stop_slice)+1,corners[2][1]+t] = [1]*(max(start_slice,stop_slice)+1 - min(start_slice,stop_slice))
        start_slice = stop_slice

    start_slice = corners[3][0]
    for t in range(1,support):
        stop_slice = corners[3][0]+int(scale*B4[t])
        mask[min(start_slice,stop_slice):max(start_slice,stop_slice)+1,corners[3][1]+t] = [1]*(max(start_slice,stop_slice)+1 - min(start_slice,stop_slice))
        start_slice = stop_slice
    
    return mask

def gen_brownian_square_mask(corners=None, mask =None, scale = 20.0, midpoint = None, width = None, img_size=512):
    if mask is None:
        mask = np.zeros(shape=(img_size, img_size))
    if corners is None:
        if midpoint is None:
            midpoint = (mask.shape[0]//2, mask.shape[1]//2)
        if width is None:
            width = mask.shape[0]//4
        corners = gen_corners(midpoint=midpoint, width = width)
        print(corners)
    mask_contour = gen_brownian_square_contour(corners, mask, scale)
    return fill_mask(mask_contour)


def gen_ellipse(img_size, center, rad_a, rad_b, R):
    mask = np.ones([img_size, img_size])
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if (center[0] - x)**2/rad_a + (center[1] - y)**2/rad_b <= R**2:
                mask[x,y] = 0
    return mask
#def gen_corners(midpoint, width):
#    return [midpoint + (width,width),midpoint + (-width,width),midpoint + (-width,-width),midpoint + (width,-width)]

def gen_corners(midpoint, width):
    return [[midpoint[0] + width,midpoint[1]+width],[midpoint[0] - width,midpoint[1]+width],[midpoint[0] - width,midpoint[1]-width],[midpoint[0] + width,midpoint[1]-width]]


# %%
if __name__ == "__main__":
    import GenTextures as gt
    import matplotlib.pylab as plt
    H = 0.7
    t = 1
    d = 2
    hfbf_f, hfbf_p = gt.gen_hfbf(512,0.5, True)
    efbf_f, efbf_p = gt.gen_efbf(512, H, t, d, True)

    brownian_mask = gen_brownian_square_mask(width = 100, scale = 25, img_size = 512)

    het_text_br = apply_smoothed_mask(hfbf_f, efbf_f, brownian_mask, filter_length = 51, sigma = 5.0)

    ellipse_mask = gen_ellipse(512, [256,256], 1, 2, 100)

    het_text_ellipse = apply_smoothed_mask(hfbf_f, efbf_f, ellipse_mask, filter_length = 11, sigma = 5.0)


    fig, axs = plt.subplots(2,2, figsize = [9,9])
    axs[0,0].imshow(brownian_mask)    
    axs[0,1].imshow(het_text_br)    
    axs[1,0].imshow(ellipse_mask)    
    axs[1,1].imshow(het_text_ellipse)    
    
# %%
    