import matplotlib.pyplot as plt
import math
import numpy as np
from utils import tool
import torch        

def add_particle(img_sz, particle):
    """
    This method cannot build very small particle image.
    """
    u, v = np.meshgrid(np.arange(0, img_sz[0]),np.arange(0, img_sz[1]))
    u, v = u[:,:,np.newaxis], v[:,:,np.newaxis]
    
    x = np.reshape(particle.x, (1,1,-1))
    y = np.reshape(particle.y, (1,1,-1))
    dp = np.reshape(particle.d, (1,1,-1))
    intensity = np.reshape(particle.i, (1,1,-1))

    image = np.exp(-8*((u-x)**2+(v-y)**2)/dp**2)*intensity # Gaussian function
    image = np.sum(image, axis=-1)*255.0
    image = image + np.random.randint(-20, 20, image.shape)
    # image  = np.round(image).astype(np.uint8) 
    image  = np.round(image)
    image = image[10:-10,10:-10]
    return image

def erf(x):
    """
    It's hard to believe we have to wrapper the erf function from pytorch
    """
    x = torch.tensor(x)
    y = torch.erf(x).cpu().numpy()
    return y

def add_particle2(img_sz, particle):
    """
    Using the erf function to synthesis the particle images
    """
    image = np.zeros(img_sz)
    v, u = np.meshgrid(np.arange(0, img_sz[0]),np.arange(0, img_sz[1]))
    # u, v = u[:,:waxis], v[:,:,np.newaxis]
    
    x_s = np.reshape(particle.x, (-1,1))
    y_s = np.reshape(particle.y, (-1,1))
    dp_s = np.reshape(particle.d, (-1,1))
    intensity_s = np.reshape(particle.i, (-1,1))
    dp_nominal=particle.nd

    for x, y, dp, intensity in zip(x_s, y_s, dp_s, intensity_s):
        ind_x1 = np.int(min(max(0, x-3*dp-2), img_sz[0]-6*dp-3))
        ind_y1 = np.int(min(max(0, y-3*dp-2), img_sz[1]-6*dp-3))
        ind_x2 = ind_x1 + np.int(6*dp+3)
        ind_y2 = ind_y1 + np.int(6*dp+3)
       
        lx = u[ind_x1:ind_x2, ind_y1:ind_y2]-x
        ly = v[ind_x1:ind_x2, ind_y1:ind_y2]-y
        b = dp/np.sqrt(8) # from the Gaussian intensity profile assumption

        img =(erf((lx+0.5)/b)-erf((lx-0.5)/b))*(erf((ly+0.5)/b)-erf((ly-0.5)/b))
        img = img*intensity  
        
        image[ind_x1:ind_x2, ind_y1:ind_y2] =  image[ind_x1:ind_x2, ind_y1:ind_y2]+ img
    
    b_n = dp_nominal/np.sqrt(8)
    partition = 1.5*(erf(0.5/b_n)-erf(-0.5/b_n))**2
    image = np.clip(image/partition,0,1.0) 
    image = image*255.0
    image  = np.round(image)
    return image

def gen_image_pair(config):
    # settings 
    img_sz = (config.img_sz[0]+20,config.img_sz[1]+20) # add boundary 
    ppp = config.ppp
    dp, d_std = config.dp, config.d_std
    i_std = config.i_std
    miss_ratio = config.miss_ratio
    displacement = config.displacement 

    # generate particles' parameters
    p1, p2= tool.AttrDict(), tool.AttrDict()
    p1.num = p2.num = np.round(ppp*np.prod(img_sz)).astype(np.int)
    p1.nd = p2.nd = dp
    p1.x = p2.x = np.random.uniform(0,img_sz[0],p1.num)
    p1.y = p2.y = np.random.uniform(0,img_sz[1],p1.num)
    p1.d = p2.d = np.abs(np.random.randn(p1.num)*d_std+ dp)
    p1.i = p2.i = np.random.randn(p1.num)*i_std+ 0.85

    p1.x = p1.x + displacement/2
    p2.x = p2.x - displacement/2
 
    # generate images
    img1 = add_particle2(img_sz,p1)
    img2 = add_particle2(img_sz,p2)
    # img1 = add_particle(img_sz,p1)
    # img2 = add_particle(img_sz,p2)

    img1=img1[10:-10,10:-10]
    img2=img2[10:-10,10:-10]
    return img1, img2

def main():
    config = tool.AttrDict
    config.img_sz = (256,256)
    config.ppp = 0.05
    config.dp = 2.2
    config.d_std = 0.1
    config.i_std =0.1
    config.miss_ratio = 0.1
    config.displacement = 2.25

    img1, img2 = gen_image_pair(config)

    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)

    plt.show()

if __name__=='__main__':
   main()	

