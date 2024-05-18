""" 
1. Find the position (x,y) that makes r(x,y) max
r_map: W*H*B
return 2*B(position), 5*B(values along x), 5*B(values along y) 

2. Find the ssubpixel 
maxvalues: 5*B
return B(subpixel estimation)
"""
import numpy as np


def argmax2d(r_map):
    # r_map: W*H*B
    # ind2d: 2*B
    # vmax: 5*1*B, 1*5*B with 5*5 neighborhood -> for subpixel estimation

    # Reshape
    W, H, B = r_map.shape
    r_flatten  = r_map.reshape(-1, B)

    # The arg max for the data W*H
    ind = np.argmax(r_flatten, axis=0)
    dx, dy = np.unravel_index(ind, [W, H])
    dx, dy = np.clip(dx,2,W-3), np.clip(dy, 2, H-3)

    # output the positions ind2d, and the value around it
    dz = np.arange(B)
    vmax_x = r_map[[dx-2,dx-1,dx,dx+1,dx+2],[dy,dy,dy,dy,dy],[dz,dz,dz,dz,dz]]
    vmax_y = r_map[[dx,dx,dx,dx,dx],[dy-2,dy-1,dy,dy+1,dy+2],[dz,dz,dz,dz,dz]]

    dx, dy  = dx-W/2.0, dy-H/2.0 # sub W/2 for displacement estimation
    ind2d = np.stack([dx,dy], axis=0)
    return ind2d, vmax_x, vmax_y


def centroid(vmax, s=5):
    # vmax: 5*B, represents [v[x-2],v[x-1],v[x],v[x+1],v[x+2]] for B batches
    # return B
    assert vmax.shape[0]==5
    assert s==3 or s==5, "the area size should be 3 or 5"
    w1 = [0,1,1,1,0] if s==3 else [1,1,1,1,1]
    w2 = [0,-1,0,1,0] if s==3 else [-2,-1,0,1,2]
    w1, w2 = np.array(w1).reshape(-1,1), np.array(w2).reshape(-1,1)

    delta_d = np.sum(vmax*w2,axis=0)/np.sum(vmax*w1,axis=0)
    return delta_d


def parabolic(vmax, s=5):
    # vmax: 5*B
    # return B
    assert vmax.shape[0]==5
    assert s==3 or s==5, "the area size should be 3 or 5"

    w1 = [0,-2,4,2,0] if s==3 else [-20,10,20,10,-20]
    w2 = [0,-1,0,1,0] if s==3 else [-14,-7,0,7,14]
    w1, w2 = np.array(w1).reshape(-1,1), np.array(w2).reshape(-1,1)

    delta_d = np.sum(vmax*w2,axis=0)/(np.sum(vmax*w1,axis=0)+1e-9)
    return delta_d


def gaussian(vmax, s=5):
    # vmax: 5*B
    # return B
    assert np.all(vmax>0), "The vmax should > 0"
    return parabolic(np.log(vmax), s)


def unit_test():
    rmap = np.random.rand(32,32,4)
    indx, vmax_x, vmax_y = argmax2d(rmap)
    for m in ["centroid", "parabolic", "gaussian"]:
        for s in [3,5]:
            dx = eval(m)(vmax_x,s)
            print(m,dx)


if __name__ == "__main__":
    unit_test()
