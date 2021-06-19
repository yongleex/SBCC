import numpy as np


def sub_pixel(r_map, win_sz, method="gaussian"):
    r_flatten = r_map.reshape(-1, r_map.shape[-1])
    r_flatten[r_flatten<1e-5] = 1e-5
    r_flatten = r_flatten*r_flatten
    # r_flatten[r_flatten<0.05] = 0.05
    # r_flatten = r_flatten + 1e-3*np.random.rand(*r_flatten.shape)
    r_flatten = r_flatten / np.max(r_flatten, axis=0, keepdims=True)
    ind = np.argmax(r_flatten , axis=0)
    dx, dy = np.unravel_index(ind, win_sz)
    dx, dy = np.clip(dx, 2, win_sz[0]-3), np.clip(dy, 2, win_sz[1]-3)

    patch_index = np.arange(dy.shape[-1])
    r_max = r_map[dx,dy,patch_index]
    r_map = r_map/np.reshape(r_max, (1,1,-1))
    r_map = np.clip(r_map, 1e-8, 1.0)
    # print(r_map.shape, r_max.shape)
    r_max = r_map[dx,dy,patch_index]
    r_r2, r_r1= r_map[dx-2,dy,patch_index], r_map[dx-1,dy,patch_index] # right
    r_l1, r_l2= r_map[dx+1,dy,patch_index], r_map[dx+2,dy,patch_index] # left
    r_u2, r_u1= r_map[dx,dy-2,patch_index], r_map[dx,dy-1,patch_index] # upper
    r_d1, r_d2= r_map[dx,dy+1,patch_index], r_map[dx,dy+2,patch_index] # down

    if method =="centroid":
        # centroid method
        u = dx + (-r_r1 + r_l1)/(r_r1+r_max+r_l1)
        v = dy + (-r_u1 + r_d1)/(r_u1+r_max+r_d1)
    elif method =="parabolic":
        # parabolic peak fit
        u = dx + (-r_r1+r_l1)/(-2*r_r1-2*r_l1+4*r_max)
        v = dy + (-r_u1+r_d1)/(-2*r_u1-2*r_d1+4*r_max)
    elif method =="gaussian":
        # Gaussian peak fit with 3 points 
        r_max = np.log(r_max.clip(1e-8,1.0))
        r_r1, r_l1 = np.log(r_r1.clip(1e-8,1)), np.log(r_l1.clip(1e-8,1))
        r_d1, r_u1 = np.log(r_d1.clip(1e-8,1)), np.log(r_u1.clip(1e-8,1))

        u = dx + (-r_r1+r_l1)/(-2*r_r1-2*r_l1+4*r_max-1e-9)
        v = dy + (-r_u1+r_d1)/(-2*r_u1-2*r_d1+4*r_max-1e-9)
    elif method =="gaussian_w":
        # Gaussian peak fit with 3 points
        r_max, r_r1, r_l1, r_d1, r_u1 = r_max*1.2-0.2, r_r1*1.2-0.2, r_l1*1.2-0.2, r_d1*1.2-0.2, r_u1*1.2-0.2
        r_max, r_r1, r_l1, r_d1, r_u1 =r_max.clip(1e-8,1.0), r_r1.clip(1e-8,1.0), r_l1.clip(1e-8,1.0), r_d1.clip(1e-8,1.0), r_u1.clip(1e-8,1.0)  
        # r_max = np.log(r_max)
        # r_r1, r_l1 = (r_r1+0.1)*np.log(r_r1), (r_l1+0.1)*np.log(r_l1)
        # r_d1, r_u1 = (r_d1+0.1)*np.log(r_d1), (r_u1+0.1)*np.log(r_u1)
        r_max = np.log(r_max)
        r_r1, r_l1 = np.log(r_r1), np.log(r_l1)
        r_d1, r_u1 = np.log(r_d1), np.log(r_u1)
        # r_max = 1-1./r_max 
        # r_r1, r_l1 = 1-1./r_r1, 1-1./r_l1
        # r_d1, r_u1 = 1-1./r_d1, 1-1./r_u1

        u = dx + (-r_r1+r_l1)/(-2*r_r1-2*r_l1+4*r_max)
        v = dy + (-r_u1+r_d1)/(-2*r_u1-2*r_d1+4*r_max)
    elif method =="gaussian5":
        # Gaussian peak fit with 5 points
        r_max = np.log(r_max.clip(1e-8,1.0))
        r_r1, r_l1 = np.log(r_r1.clip(1e-8,1)), np.log(r_l1.clip(1e-8,1))
        r_r2, r_l2 = np.log(r_r2.clip(1e-8,1)), np.log(r_l2.clip(1e-8,1))
        r_d1, r_u1 = np.log(r_d1.clip(1e-8,1)), np.log(r_u1.clip(1e-8,1))
        r_d2, r_u2 = np.log(r_d2.clip(1e-8,1)), np.log(r_u2.clip(1e-8,1))

        u = dx + (14*r_r2+7*r_r1 -7*r_l1-14*r_l2)/(20*r_r2-10*r_r1 -20*r_max -10*r_l1 +20*r_l2)
        v = dy + (14*r_u2+7*r_u1 -7*r_d1-14*r_d2)/(20*r_u2-10*r_u1 -20*r_max -10*r_d1 +20*r_d2)
    elif method is None:
        u , v = dx, dy
    else:
        u , v = dx, dy
        raise ValueError('Unknown sub-pixel method')
    u = u-win_sz[0]//2 
    v = v-win_sz[1]//2
    return u,v

