import cv2
import numpy as np


def context_min(images):
    return [np.min(images, axis=0)]

def context_mean(images):
    return [np.mean(images, axis=0)]

def context_blurU(images, kx=21, ky=21):
    results = [cv2.blur(img, (kx,ky)) for img in images]
    return results 
    
def context_blurG(images, kx=21, ky=21):
    results = [cv2.GaussianBlur(img, (kx,ky), 3.0) for img in images]
    return results 

def test():
    import matplotlib.pyplot as plt

    img1 = cv2.imread('./data/3a.tif',0)
    img2 = cv2.imread('./data/3b.tif',0)
    images = [img1, img2]

    for func in [context_min, context_mean, context_blurU, context_blurG]:
        bgs = func(images)
        for bg in bgs:
            plt.figure()
            plt.imshow(bg)
            plt.title(func.__name__)
    plt.show()

if __name__ == "__main__":
    test() 
