import cv2
import numpy as np
import seaborn as sns
sns.set_style()

"""
cross-correlation methods
"""

def scc(img1, img2):
    """ standard fft based cross correlation"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = F1*np.conj(F2)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r


def pc(img1, img2):
    """ phase correlation"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = F1*np.conj(F2)
    R = R/(np.abs(R)+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r


def sbcc_pc(img1, img2):
    """ modified phase correlation with SBCC"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    deno = 0.5*(F1*np.conj(F1)+F2*np.conj(F2))+1e-5
    R = F1*np.conj(F2)/deno
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r


def spof(img1, img2):
    """ symmetric phase-only filter"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = F1*np.conj(F2)
    R = R/(np.sqrt(np.abs(R))+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r

def rpc(img1, img2, var=2.25, de=2.2):
    """robust phase correlation"""
    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
    x, y = x-img1.shape[0]//2 , y- img1.shape[1]//2 
    # var = de**2/8 # This is equivalent to 8/de**2 in frequency domain.
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G = np.fft.rfft2(np.fft.ifftshift(shift_g))[:,:,np.newaxis]
    # print(G)

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = F1*np.conj(F2)
    R = G*R/(np.abs(R)+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    # shift_r =  shift_r*0 + shift_g[:,:,np.newaxis]
    return shift_r


def sbcc_rpc(img1, img2, var=2.25, de=2.2):
    """modified robust phase correlation with SBCC"""
    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
    x, y = x-img1.shape[0]//2 , y- img1.shape[1]//2 
    # var = de**2/8 # This is equivalent to 8/de**2 in frequency domain.
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G = np.fft.rfft2(np.fft.ifftshift(shift_g))[:,:,np.newaxis]
    # print(G)

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    deno = 0.5*(F1*np.conj(F1)+F2*np.conj(F2))+1e-5
    R = F1*np.conj(F2)
    R = G*R/deno
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    # shift_r =  shift_r*0 + shift_g[:,:,np.newaxis]
    return shift_r


def cfcc(img1, img2, var=2.25,lamda=0.1):
    """ correlation-filter based cross correlation
        var: the variance for gaussian response
        eps: a regularization parameter, recommended to be 0.1 [MOSSE]
    """
    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
    x, y = x-img1.shape[0]//2, y- img1.shape[1]//2 
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G = np.fft.rfft2(np.fft.ifftshift(shift_g))[:,:,np.newaxis]

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = F1*G*np.conj(F2)/(F2*np.conj(F2)+lamda) 
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r


def sbcc(img1, img2, var=2.25, lamda=1e-5, mu=1.0, nu=10.0):
    """surrogate-based cross correlation"""
    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
    x, y = x-img1.shape[0]//2, y-img1.shape[1]//2 
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G =np.real(np.fft.rfft2(np.fft.ifftshift(shift_g))[:,:,np.newaxis])
    Gc = np.conj(G)
    Gd_p = np.sqrt(G) # =Gd*np.conj(Gd)

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    F1c, F2c = np.conj(F1), np.conj(F2)
    A1, A2 = F1*F1c, F2*F2c
    Num = A1.shape[-1]
    if Num==1:
        nu = 0
    # PN = (np.sum(G*Gc*A1+G*Gc*A2, axis=-1, keepdims=True)-(G*Gc*A1+G*Gc*A2))/(Num-1+1e-8) # the mean of neg power 
    PN = (np.sum(A1+A2, axis=-1, keepdims=True)-(A1+A2))/(Num-1+1e-8) # the mean of neg power 
    # print(np.max(PN), np.min(PN))

    # nume = 2*(G+mu*np.sqrt(G))*F1*F2c
    # deno = A1 + A2 + 2*lamda + 2*mu*np.sqrt(G) + 2*nu*PN*np.sqrt(G)
    nume = 2*(G+mu*Gd_p)*F1*F2c
    deno = A1 + A2 + 2*lamda + 2*mu*Gd_p + 2*nu*PN*np.sqrt(G)
    # R = nume/deno
    R = (1+1.5*nu)* nume/deno # The (1+nu) is a constant, to make the max r close to 1.0
    r = np.fft.irfft2(R, axes=(0,1)) 
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r


def sbcc_b1(img1, img2):
    return sbcc(img1, img2, var=2.25, lamda=1e-5, mu=0.0, nu=0.0)


def sbcc_b2(img1, img2):
    return sbcc(img1, img2, var=2.25, lamda=0.1, mu=0.0, nu=0.0)


def sbcc_b3(img1, img2):
    return sbcc(img1, img2, var=2.25, lamda=1e-5, mu=1.0, nu=0.0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plot import plot_cc_map 
    def add_noise(img):
        img = img+ np.random.normal(0, 10, img.shape)
        return img

    def preprocessing(img):
        img = img- np.mean(img)
        img = img/np.linalg.norm(img)
        return img

        
    #- Visualize CC methods correlation map
    image1 = cv2.imread('./TestImages/img1.png', 0)[:,:,np.newaxis]
    image2 = cv2.imread('./TestImages/img2.png', 0)[:,:,np.newaxis]
    # image1 = cv2.imread('./TestImages/2a.tif', 0)[:,:,np.newaxis]
    # image2 = cv2.imread('./TestImages/2b.tif', 0)[:,:,np.newaxis]
    b_sz = 00 # 80, 200
    w_sz = 32
    image1 = image1[b_sz:b_sz+w_sz,b_sz:b_sz+w_sz]
    image2 = image2[b_sz:b_sz+w_sz,b_sz:b_sz+w_sz] # well paired
    # image2 = image2[12:12+w_sz,10:10+w_sz] # large displacement w.r.t window size
    # image2 = image2[25:25+w_sz,:w_sz] # out of range
    image1 , image2 = preprocessing(image1) , preprocessing(image2)
    # image2 = image1

    methods =['scc','pc', 'spof', 'rpc','cfcc', 'sbcc', 'sbcc_b1', 'sbcc_b2']
    for method in methods:
        r = eval(method)(image1, image2)
        plot_cc_map(r[:,:,0], method.upper())
    plt.show()
