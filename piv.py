import numpy as np
import cv2
from utils.win import grid_window
from utils.cc import pc, spof, rpc
from utils.cc import sbcc_pc, sbcc_rpc
from utils.cc import scc, cfcc, sbcc, sbcc_b1, sbcc_b2, sbcc_b3 
from utils.subpixel import sub_pixel
from utils.plot import plot_cc_map
import matplotlib.pyplot as plt
from utils import tool, outlier

class PIV:
    def __init__(self, config):
        self._c = config

    def grid_window(self, img):
        win, vec_shape = grid_window(img, step_sz=self._c.step_sz, win_sz=self._c.win_sz)
        return win, vec_shape

    def compute_onepass(self, image1, image2):
        # divide the window
        # win1, vec_shape = grid_window(image1, step_sz=self._c.step_sz, win_sz=self._c.win_sz)
        # win2, _ = grid_window(image2, step_sz=self._c.step_sz, win_sz=self._c.win_sz)
        win1, vec_shape = self.grid_window(image1)
        win2, _ = self.grid_window(image2)

        # a basic pre-processing
        win1 = self.preprocessing(win1)
        win2 = self.preprocessing(win2)

        # win1 = self.pad_zero(win1, self._c.win_sz[0]//2)
        # win2 = self.pad_zero(win2, self._c.win_sz[0]//2)

        # obtain correlation coefficients map
        r_map = eval(self._c.method)(win1, win2)
        # plot_cc_map(r_map[:,:,0])
        

        # sub-pixel peak interpolation
        # win_sz = [self._c.win_sz[0]*2, self._c.win_sz[1]*2]
        win_sz = [self._c.win_sz[0], self._c.win_sz[1]]
        u, v = sub_pixel(r_map, win_sz=win_sz, method=self._c.subpixel)
        u, v = u.reshape(vec_shape, order='C'), v.reshape(vec_shape, order='C')
        return u, v, r_map

    def compute(self, image1, image2):
        u, v, r_map = self.compute_onepass(image1, image2)
        
        img1, img2 = image1.copy(), image2.copy()
        for _ in range(self._c.runs-1): # Multi-pass window deformation method
            # remove the outliers and to dense motion field
            u, v, _ = outlier.NMT(u,v)

            x, y = np.meshgrid(np.arange(0, image1.shape[1]), np.arange(0, image1.shape[0]))
            x_ind =  (x-self._c.win_sz[0]/2)/self._c.step_sz[0]
            y_ind =  (y-self._c.win_sz[1]/2)/self._c.step_sz[1]
            
            u_dense = 0.5*self.remap(u, y_ind, x_ind)
            v_dense = 0.5*self.remap(v, y_ind, x_ind)

            # wrapping the images
            # cv2.imwrite('1.png', image1)
            # cv2.imwrite('4.png', image2)
            image1 = self.remap(img1, x+u_dense, y+v_dense)
            image2 = self.remap(img2, x-u_dense, y-v_dense)
            # cv2.imwrite('2.png', image1)
            # cv2.imwrite('3.png', image2)
            du, dv, r_map = self.compute_onepass(image1, image2)
            u, v = u+du, v+dv

        return u, v, r_map

    
    def preprocessing(self, win):
        win=win- np.mean(win, axis=(0,1), keepdims=True) +1e-3
        win=win/(np.linalg.norm(win, axis=(0,1), keepdims=True)+1e-8)
        return win

    def pad_zero(self, win, sz):
        win = np.pad(win, ((sz,sz),(sz,sz),(0,0)),'constant', constant_values=(0.0))
        return win

    def remap(self, img, x, y):
        x, y = np.float32(x), np.float32(y)
        out = cv2.remap(img, x, y, cv2.INTER_CUBIC)
        return out

def main():
    # config for piv 
    config = tool.AttrDict()
    config.win_sz = [32,32]
    config.step_sz =[16,16]
    config.subpixel= "gaussian" #"parabolic" # 'centroid' # 'gaussian'
    config.method='cfcc3'
    config.runs = 3

    piv = PIV(config)

    image1 = cv2.imread('./TestImages/3a.tif', 0)
    image2 = cv2.imread('./TestImages/3b.tif', 0)
    # image1 = cv2.imread('./TestImages/1a.png', 0)
    # image2 = cv2.imread('./TestImages/1b.png', 0)

    u, v, r_map= piv.compute(image1, image2)
    # u, v, r_map= piv.multi_pass_compute(image1, image2)

    # debug part
    from utils.plot import plot_cc_map
    plot_cc_map(r_map[:,:,100], config.method)

    plt.figure()
    plt.imshow(image1)
    x, y = np.meshgrid(np.arange(0, u.shape[0]), np.arange(0, u.shape[1]))
    fig = plt.figure()
    ax = fig.gca()
    ax.invert_yaxis()
    ax.quiver(-u.transpose(),v.transpose())
    # ax.quiver(u,v)
    plt.show()

if __name__ == '__main__':
    main()
