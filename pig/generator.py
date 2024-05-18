import numpy as np
from scipy.special import erf

from flow import displacement, flow_dict

class Generator():
    def __init__(self, cfg):
        self._c = cfg
        
        def flow(x,y):
            param = getattr(self._c, self._c.flow)
            return flow_dict[self._c.flow](x, y, **param) 
        self.flow = flow

    def compute(self):
        # generate random particles
        number = self._c.ppp*(self._c.img_sz[0]+2*self._c.img_bd)*(self._c.img_sz[1]+2*self._c.img_bd)
        number = np.int64(number)

        xs1 = np.random.rand(number)*(self._c.img_sz[0]+2*self._c.img_bd)-self._c.img_bd
        ys1 = np.random.rand(number)*(self._c.img_sz[1]+2*self._c.img_bd)-self._c.img_bd
        xs2, ys2 = displacement(self.flow,xs1,ys1)

        dp_s = self._c.sigma_d * np.random.randn(number) + self._c.mu_d
        int_s = self._c.sigma_i * np.random.randn(number) + self._c.mu_i
        
        img1 = self._one_img(self._c.img_sz, xs1, ys1, dp_s, int_s, d_m=self._c.mu_d)
        img2 = self._one_img(self._c.img_sz, xs2, ys2, dp_s, int_s, d_m=self._c.mu_d)

        # generate the truth
        x, y = np.meshgrid(np.arange(self._c.img_sz[0]),np.arange(self._c.img_sz[1]), indexing="ij")
        ut, vt = self.flow(x,y)
        return img1, img2, ut, vt

    def _one_img(self, img_sz, x_s, y_s, dp_s, int_s, d_m=2.5):
        """
        Using the erf function to synthesis the particle images
        """
        image = np.zeros(img_sz)
        u, v = np.meshgrid(np.arange(img_sz[0]),np.arange(img_sz[1]), indexing="ij")
        
        for x, y, dp, intensity in zip(x_s, y_s, dp_s, int_s):
            ind_x1 = int(min(max(0, x-3*dp-2), img_sz[0]-6*dp-3))
            ind_y1 = int(min(max(0, y-3*dp-2), img_sz[1]-6*dp-3))
            ind_x2 = ind_x1 + int(6*dp+3)
            ind_y2 = ind_y1 + int(6*dp+3)
           
            lx = u[ind_x1:ind_x2, ind_y1:ind_y2] -x
            ly = v[ind_x1:ind_x2, ind_y1:ind_y2] -y
            b = dp/np.sqrt(8) # from the Gaussian intensity profile assumption
    
            img =(erf((lx+0.5)/b)-erf((lx-0.5)/b))*(erf((ly+0.5)/b)-erf((ly-0.5)/b))
            img = img*intensity
            
            image[ind_x1:ind_x2, ind_y1:ind_y2] =  image[ind_x1:ind_x2, ind_y1:ind_y2]+ img
        
        b_n = d_m/np.sqrt(8)
        partition = 1.5*(erf(0.5/b_n)-erf(-0.5/b_n))**2
        image = np.clip(image/partition,0,1.0) 
        image = image*255.0
        image  = np.round(image)
        return image

