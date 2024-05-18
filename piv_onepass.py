import numpy as np
import cv2

from config import config
from estimator import CrossCorr, CrossCorr2
from background import context_min, context_blurU, context_blurG

def piv_scc(img1, img2, cfg = None):
    if cfg is None:
        cfg = config()
    cfg.cc.cc_method= "scc"
    estimator = CrossCorr(cfg.cc)
    x1,y1,u1,v1 = estimator.compute(img1, img2)
    return x1, y1, u1, v1

def piv_scc_min(img1, img2, cfg = None):
    bkg = context_min([img1, img2])[0]
    return piv_scc(img1-bkg,img2-bkg, cfg=cfg)

def piv_scc_lpf(img1, img2, cfg = None):
    bkg = context_blurG([img1, img2])
    return piv_scc(img1-bkg[0],img2-bkg[1], cfg=cfg)

def piv_spof(img1, img2, cfg = None):
    if cfg is None:
        cfg = config()
    cfg.cc.cc_method= "spof"
    estimator = CrossCorr(cfg.cc)
    x1,y1,u1,v1 = estimator.compute(img1, img2)
    return x1, y1, u1, v1
    
def piv_rpc(img1, img2, cfg = None):
    if cfg is None:
        cfg = config()
    cfg.cc.cc_method= "rpc"
    estimator = CrossCorr(cfg.cc)
    x1,y1,u1,v1 = estimator.compute(img1, img2)
    return x1, y1, u1, v1

def piv_sbcc(img1, img2, contexts=None, mu=3.0, cfg = None):
    if cfg is None:
        cfg = config()
            
    estimator = CrossCorr2(cfg.cc)
    if contexts is None:
        contexts = context_min([img1, img2])
        contexts.append(context_blurG([img1, img2])[0])
        contexts.append(context_blurG([img1, img2])[1])
    x1,y1,u1,v1 = estimator.compute(img1, img2, contexts, mu=mu)
    return x1, y1, u1, v1


def test():
    import matplotlib.pyplot as plt
    from utils.metrics import  PIVmetric
    
    def add_background(img1, img2):
        sz = img1.shape
        x, y = np.meshgrid(np.linspace(0,sz[0], sz[0]), np.linspace(0,sz[1], sz[1]), indexing="ij")
        # bkg = 50*np.sin(2*np.pi*x/8)+50*np.sin(2*np.pi*y/8)+100
        # bkg = 100*(np.cos(2*np.pi*y/32)>0)*(np.cos(2*np.pi*x/32)>0)+20.*np.sin(2*np.pi*y/5)+25
        bkg = 128*(np.cos(2*np.pi*y/12)>0)
        
        img1_ = img1+bkg+0.0
        img2_ = img2+bkg+0.0
        return img1_, img2_
        
    img1 = cv2.imread('./data/vp1a.tif',0)
    img2 = cv2.imread('./data/vp1b.tif',0)
    img1_, img2_ = add_background(img1, img2)
    
    plt.figure(figsize=(10,10))
    names = ["image1", "image2", "image1+bg", "image2+bg"]
    for k, img in enumerate([img1, img2, img1_, img2_]):
        plt.subplot(2,2,k+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(names[k])
        
    plt.figure(figsize=(15,10))
    xt,yt,ut,vt = piv_scc(img1, img2)
    for k, m in enumerate([piv_scc, piv_scc_min, piv_scc_lpf, piv_spof, piv_rpc, piv_sbcc]):
        x1,y1,u1,v1 = m(img1_, img2_)
        res = PIVmetric(ut,vt,u1,v1)
        plt.subplot(2,3,k+1)
        plt.quiver(x1,y1,u1,v1) # Without any modification
        plt.title(m.__name__)
    plt.show()

if __name__=="__main__":
    test()
