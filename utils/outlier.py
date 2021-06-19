import numpy as np
import cv2


def NMT(u,v, eps=0.2, thr=5.0, smooth_flag=True):
    """ 
    Normalised Median Test, from 'Universal outlier detection for PIV data'
    """
    u, v = np.float32(u), np.float32(v)
    criterion = 0
    
    for c in [u,v]:
        c_median = cv2.medianBlur(c, 5)
        residual = np.abs(c - c_median)
        r_median = cv2.medianBlur(residual, 5)
        cri = residual/(r_median + eps)
        criterion += np.power(cri, 2)

    criterion = np.sqrt(criterion)
    index = criterion > thr

    u_out, v_out = u, v
    u_out[index] = cv2.medianBlur(u, 5)[index]
    v_out[index] = cv2.medianBlur(v, 5)[index]
    
    if smooth_flag:
        u_out = cv2.GaussianBlur(u_out, (3,3),0)
        v_out = cv2.GaussianBlur(v_out, (3,3),0)
    return u_out, v_out, index



