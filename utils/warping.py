import numpy as np
from scipy import ndimage


def remap(f,x,y): # an improved implementation for remap in OpenCV
    # output f(x,y) as the shape of array x or array y
    # f is the value in the gridded positions
    out =ndimage.map_coordinates(f,(x,y), order=5, mode="nearest", prefilter=False)
    return out


def sparse2dense(x, y, u, v, sz):
    den_x, den_y = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing="ij")
    sample_x = (den_x-x[0,0])/(x[1,0]-x[0,0])
    sample_y = (den_y-y[0,0])/(y[0,1]-y[0,0])
    den_u = remap(u, sample_x, sample_y)
    den_v = remap(v, sample_x, sample_y)
    return den_x, den_y, den_u, den_v


""" image warping with deformation field or velocity field.
"""
def ss(u, v, delta=1, n_iter=9):
    """ Generate deformation field
    Integrates a vector field via scaling and squaring.
    adopted from https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """
    assert u.shape == v.shape

    x, y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing="ij")
    u, v = u/delta, v/delta
    dx, dy = u/2**n_iter, v/2**n_iter

    for iter in range(n_iter):
        dx_new = dx + remap(dx, x+dx, y+dy)
        dy_new = dy + remap(dy, x+dx, y+dy)
        dx, dy = dx_new, dy_new

    dx, dy = dx*delta, dy*delta
    return dx, dy


def warp(img1, img2, u, v, method='CDI', t=0.5):
    # FDI: out1(x,y)=img1(x+u, y+v)
    # FDI, CDI, FDDI, CDDI

    assert img1.shape == img2.shape == u.shape == v.shape
    assert method in ["FDI", "FDDI", "FDI2", "FDDI2", "CDI", "CDDI"]

    x, y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing="ij")
    if method == 'FDI':
        out1= remap(img1, x-u, y-v)
        out2= remap(img2, x, y)
        # out2= img2.copy()
    elif method == 'FDI2':
        # out1= img1.copy()
        out1= remap(img1, x, y)
        out2= remap(img2, x+u, y+v)
    elif method == 'CDI':
        out1= remap(img1, x-0.5*u, y-0.5*v)
        out2= remap(img2, x+0.5*u, y+0.5*v)
    elif method == 'FDDI':
        dx, dy = ss(-u, -v, delta=1)
        out1= remap(img1, x+dx, y+dy)
        # out2= img2.copy()
        out2= remap(img2, x, y)
    elif method == 'FDDI2':
        dx, dy = ss(u, v, delta=1)
        # out1= img1.copy()
        out1= remap(img1, x, y)
        out2= remap(img2, x+dx, y+dy)
    elif method == 'CDDI':
        dx1, dy1 = ss(-t*u, -t*v, delta=1)
        dx2, dy2 = ss((1-t)*u, (1-t)*v, delta=1)
        out1= remap(img1, x+dx1, y+dy1)
        out2= remap(img2, x+dx2, y+dy2)
    else:
        raise NotImplementedError
    return out1, out2

def unit_test_remap():
    import cv2
    import matplotlib.pyplot as plt
    
    img0 = cv2.imread("./img.png", 0)
    x, y = np.meshgrid(np.arange(img0.shape[0]), np.arange(img0.shape[1]), indexing="ij")

    img1 = remap(img0, x-100, y)
    img2 = remap(img0, x, y-100)
    plt.figure(figsize=(15,5))
    plt.title("Test remap func")
    plt.subplot(131); plt.imshow(img0, cmap="gray"); plt.axis("off");
    plt.subplot(132); plt.imshow(img1, cmap="gray"); plt.axis("off");
    plt.subplot(133); plt.imshow(img2, cmap="gray"); plt.axis("off");
    plt.show()

if __name__ == "__main__":
    unit_test_remap()

