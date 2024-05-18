"""
One-pass estimators for PIV analysis
"""
import numpy as np
from scipy import ndimage

# cross correlation
from utils.win import grid_window
from utils.cc import scc, rpc, spof, sbcc
from utils.ccmap import centroid, parabolic, gaussian, argmax2d
from utils.warping import remap, sparse2dense, warp

# optical flow
import cv2

# deep neural networks (UnLiteFlowNet)
# from UnFlowNet.models import Network, device, estimate
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class Estimator():
    def __init__(self, config):
        self._c = config
        self.p = 0

    def setpass(self, p):
        self.p = p

    def compute(self, img1, img2):
        raise NotImplementedError



# cross-correlation
class CrossCorr(Estimator):
    def preprocessing(self, win):
        win=win- np.mean(win, axis=(0,1), keepdims=True)
        win=win/(np.linalg.norm(win, axis=(0,1), keepdims=True)+1e-8)
        return win

    def compute(self, img1, img2, step_sz=None, win_sz=None):
        assert img1.shape==img2.shape, "Two input images should have the same size"
        assert len(img1.shape)==2, "Only monocolor image is supported in this version"

        if step_sz is None: step_sz = self._c.step_sz[self.p]
        if win_sz is None: win_sz = self._c.win_sz[self.p]

        win1, posi, shape = grid_window(image=img1,step_sz=step_sz,win_sz=win_sz)
        win2, _, _ = grid_window(image=img2,step_sz=step_sz,win_sz=win_sz)
        if self._c.pre_norm:
            win1, win2 = self.preprocessing(win1), self.preprocessing(win2)

        r_map = eval(self._c.cc_method)(win1, win2)
        r_map = np.maximum(r_map, 1e-3) # ignore the value closed to 0

        ind2d, vmax_x, vmax_y = argmax2d(r_map)
        subx, suby = eval(self._c.sub_method)(vmax_x), eval(self._c.sub_method)(vmax_y)
        sub = np.stack([subx, suby], axis=0)

        # change the max_ind to velocity vectors 2*B
        vec = ind2d+sub

        # reshape for plot
        x, y = np.split(posi,2,axis=0)
        u, v = np.split(vec,2,axis=0)
        x, y, u, v = x.reshape(shape), y.reshape(shape), u.reshape(shape), v.reshape(shape)
        return x,y,u,v


# cross-correlation
class CrossCorr2(Estimator):
    def preprocessing(self, win):
        # print(win.shape)
        win=win- np.mean(win, axis=(0,1), keepdims=True)
        win=win/(np.linalg.norm(win, axis=(0,1), keepdims=True)+1e-9)
        return win

    def compute(self, img1, img2, bkgs, step_sz=None, win_sz=None, mu=1.0):
        assert img1.shape==img2.shape, "Two input images should have the same size"
        assert len(img1.shape)==2, "Only monocolor image is supported in this version"

        if step_sz is None: step_sz = self._c.step_sz[self.p]
        if win_sz is None: win_sz = self._c.win_sz[self.p]

        win1, posi, shape = grid_window(image=img1,step_sz=step_sz,win_sz=win_sz)
        win2, _, _ = grid_window(image=img2,step_sz=step_sz,win_sz=win_sz)
        win_bkgs = [grid_window(image=c,step_sz=step_sz,win_sz=win_sz)[0] for c in bkgs]
        if self._c.pre_norm:
            win1, win2 = self.preprocessing(win1), self.preprocessing(win2)
            win_bkgs = [self.preprocessing(win) for win in win_bkgs]

        r_map = sbcc(win1, win2, win_bkgs, mu=mu)
        r_map = np.maximum(r_map, 1e-3) # ignore the value closed to 0

        ind2d, vmax_x, vmax_y = argmax2d(r_map)
        subx, suby = eval(self._c.sub_method)(vmax_x), eval(self._c.sub_method)(vmax_y)
        sub = np.stack([subx, suby], axis=0)

        # change the max_ind to velocity vectors 2*B
        vec = ind2d+sub

        # reshape for plot
        x, y = np.split(posi,2,axis=0)
        u, v = np.split(vec,2,axis=0)
        x, y, u, v = x.reshape(shape), y.reshape(shape), u.reshape(shape), v.reshape(shape)
        return x,y,u,v

# optical flow with OpenCV
class OpticalFlow(Estimator):
    def compute(self, img1, img2, level=4):
        flow1 = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, level, 33, 11, 9, 1.3, 0)
        flow2 = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, level, 33, 11, 9, 1.3, 0)
        flow = (flow1-flow2)/2
        v, u = flow[...,0], flow[...,1]
        x, y = np.meshgrid(np.arange(u.shape[0]), np.arange(u.shape[1]), indexing="ij")
        return x,y,u,v


# # deep PIV with UnLiteFlowNet
# class LiteFlowNet(Estimator):
#     def __init__(self, cfg):
#         super().__init__(cfg)
# 
#         self.model = Network()
#         path= './UnFlowNet/UnsupervisedLiteFlowNet_pretrained.pt'
#         self.model.load_state_dict(torch.load(path)['model_state_dict'])
#         self.model.eval()
#         self.model.to(device)
#         print('unliteflownet load successfully.')
# 
#     def compute(self, img1, img2):
#         # The input of the network is recommended to be (256, 256)
#         assert img1.shape == img2.shape #== (256,256)
#         sz = img1.shape
#         h, w = sz[0], sz[1]
#         x1 = torch.Tensor(img1/255.0).view(1,1,sz[0],sz[1])
#         x2 = torch.Tensor(img2/255.0).view(1,1,sz[0],sz[1])
# 
#         if img1.shape[0] != 256 or img1.shape[1] != 256:
#             s = 16
# 
#             # padding zero to 256+n*(256-2s) >= h+2s
#             nx, ny = (h+2*s-256-1)//(256-2*s)+2, (w+2*s-256-1)//(256-2*s)+2
#             px = 256+(nx-1)*(256-2*s)-h
#             py = 256+(ny-1)*(256-2*s)-w
# 
#             pad_x1 = F.pad(x1, (s, py-s, s, px-s), "constant", 0)
#             pad_x2 = F.pad(x2, (s, py-s, s, px-s), "constant", 0)
#             hp1, wp1 = pad_x1.shape[2], pad_x1.shape[3]
#         #     print("padding size", pad_x1.shape)
# 
#             # change the shape to B*1*256*256, reorganise the large image to patches
#             win_x1 = nn.Unfold(kernel_size=(256, 256), stride=((256-2*s), (256-2*s)))(pad_x1)
#             win_x1 = win_x1.permute(2, 0, 1)  # B*N*(w*h)
#             win_x1 = win_x1.reshape(win_x1.shape[0], win_x1.shape[1], 256, 256)  # B*N*w*h
#             win_x2 = nn.Unfold(kernel_size=(256, 256), stride=((256-2*s), (256-2*s)))(pad_x2)
#             win_x2 = win_x2.permute(2, 0, 1)  # B*N*(w*h)
#             win_x2 = win_x2.reshape(win_x2.shape[0], win_x2.shape[1], 256, 256)  # B*N*w*h
#         #     print("win_x1.shape", win_x1.shape, x1.shape)
# 
#             with torch.no_grad():
#                 torch.cuda.empty_cache()
#                 # y_pre = estimate(win_x1.to(device), win_x2.to(device), unliteflownet, train=False)
#                 y_pre = estimate(win_x1.to(device), win_x2.to(device), self.model, train=False)
# 
#             y_pre = y_pre[:, :, s:-s, s:-s]
#             # change back
#             sz = (nx, ny, 2, y_pre.shape[2], y_pre.shape[3])
#             y_pre = y_pre.reshape(sz)
#             y_pre = y_pre.permute(2, 0, 3, 1, 4) # B*2*256*256-> 2*B*256*256
#             y_pre = y_pre.reshape(2, (256-2*s)*nx, (256-2*s)*ny)
# 
#             y_pre = y_pre.detach().numpy()
#             u, v = y_pre[0,:,:], y_pre[1,:,:]
#             u = u[:h, :w]
#             v = v[:h, :w]
#             assert u.shape == img1.shape # check the shape
#         else:
#             with torch.no_grad():
#                 torch.cuda.empty_cache()
#                 y_pre = estimate(x1.to(device), x2.to(device), unliteflownet, train=False)
#             u = y_pre[0][0].detach().numpy()
#             v = y_pre[0][1].detach().numpy()
# 
#         # x, y = np.meshgrid(np.arange(img1.shape[1]), np.arange(img1.shape[0]))
#         # return x, y, u, v
#         x, y = np.meshgrid(np.arange(img1.shape[0]), np.arange(img1.shape[1]), indexing="ij")
#         return x, y, v, u


def test_cc():
    import matplotlib.pyplot as plt
    from config import config

    cfg = config()
    estimator = CrossCorr(cfg.cc)

    img1 = cv2.imread("./data/vp1a.tif", 0)
    img2 = cv2.imread("./data/vp1b.tif", 0)

    x1,y1,u1,v1 = estimator.compute(img1, img2)
    plt.figure()
    plt.quiver(x1,y1,u1,v1) # Without any modification
    plt.title("The one pass result of CC")
    plt.show()


def test_of():
    import matplotlib.pyplot as plt
    from config import config

    cfg = config()
    estimator = OpticalFlow(cfg.of)

    img1 = cv2.imread("./data/vp1a.tif", 0)
    img2 = cv2.imread("./data/vp1b.tif", 0)

    x1,y1,u1,v1 = estimator.compute(img1, img2)
    plt.figure()
    # plt.quiver(x1,y1,u1,v1) # Without any modification
    plt.quiver(x1[::8,::8],y1[::8,::8],u1[::8,::8],v1[::8,::8])
    plt.title("The one pass OF")
    plt.show()


# def test_dl():
#     import matplotlib.pyplot as plt
#     from config import config
# 
#     cfg = config()
#     estimator = LiteFlowNet(cfg.dl)
# 
#     img1 = cv2.imread("./data/vp1a.tif", 0)
#     img2 = cv2.imread("./data/vp1b.tif", 0)
# 
#     x1,y1,u1,v1 = estimator.compute(img1, img2)
#     plt.figure()
#     # plt.quiver(x1,y1,u1,v1) # Without any modification
#     plt.quiver(x1[::8,::8],y1[::8,::8],u1[::8,::8],v1[::8,::8])
#     plt.title("The one pass UnFlowNet")
#     plt.show()


if __name__ == "__main__":
    test_cc()
    test_of()
    # test_dl()
