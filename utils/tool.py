import numpy as np
from scipy.ndimage import maximum_filter


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def signal2noise(r_map):
    """ Compute the signal-to-noise ratio of correlation plane.
    w*h*c"""
    r = r_map.copy()
    max_r = maximum_filter(r_map, (5,5,1))
    ind = max_r> (r_map+1e-3) 

    r[ind] = 0.05
    r = np.reshape(r, (-1, r.shape[-1]))
    r = np.sort(r,axis=0)
    ratio = r[-1,:]/r[-2,:]
    return ratio

def main():
    r = np.random.randn(5,5,3)
    signal2noise(r)


if __name__=='__main__':
    main()


