"""
cross-correlation methods
img1, img2: W*H*B
return: W*H*B for the response
"""
import cv2
import numpy as np

def scc(img1, img2):
    """ standard fft based cross correlation"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = np.conj(F1)*F2
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r

def pc(img1, img2):
    """ phase correlation"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = np.conj(F1)*F2
    R = R/(np.abs(R)+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r

def spof(img1, img2):
    """ symmetric phase-only filter"""
    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = np.conj(F1)*F2
    R = R/(np.sqrt(np.abs(R))+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r

def rpc(img1, img2, var=2.25, de=2.2):
    """robust phase correlation"""
    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[1],1), indexing='ij')
    x, y = x-img1.shape[0]//2 , y- img1.shape[1]//2 
    # var = de**2/8 # This is equivalent to 8/de**2 in frequency domain.
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G = np.fft.rfft2(np.fft.ifftshift(shift_g))
    if len(img1.shape)==3:
        G = G[:,:,np.newaxis]
    # print(G)

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    R = np.conj(F1)*F2
    R = G*R/(np.abs(R)+1e-6)
    # R = R/(np.abs(R)+1e-5)
    r = np.fft.irfft2(R, axes=(0,1))
    shift_r = np.fft.fftshift(r, axes=(0,1))
    # shift_r =  shift_r*0 + shift_g[:,:,np.newaxis]
    return shift_r

def sbcc(img1, img2, contexts=None, var=2.25, mu=3.0):
    """surrogate-based cross correlation"""
    Q = 0.0
    for k,c in enumerate(contexts):
        Fc= np.fft.rfft2(c, axes=(0,1))
        Qi = Fc*np.conj(Fc)
        Q = (k*Q+Qi)/(k+1)

    x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
    x, y = x-img1.shape[0]//2, y-img1.shape[1]//2 
    shift_g = np.exp(-0.5*(x**2+y**2)/var)
    G =np.real(np.fft.rfft2(np.fft.ifftshift(shift_g)))
    if len(img1.shape)==3:
        G = G[:,:,np.newaxis]
    Gc = np.conj(G)

    F1 = np.fft.rfft2(img1, axes=(0,1))
    F2 = np.fft.rfft2(img2, axes=(0,1))
    F1c, F2c = np.conj(F1), np.conj(F2)
    A1, A2 = F1*F1c, F2*F2c

    nume = 2*G*F1c*F2
    # nume = 2*F1c*F2
    deno = A1 + A2 + mu*Q + 1e-6
    # deno = A1 + A2 + lamda
    R = nume/deno
    r = np.fft.irfft2(R, axes=(0,1)) 
    shift_r = np.fft.fftshift(r, axes=(0,1))
    return shift_r

# def sbcc2(img1, img2, var=2.25, lamda=1e-5, mu=1.0, nu=10.0):
#     """surrogate-based cross correlation"""
#     x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[1],1), indexing='ij')
#     x, y = x-img1.shape[0]//2, y-img1.shape[1]//2 
#     shift_g = np.exp(-0.5*(x**2+y**2)/var)
#     G =np.real(np.fft.rfft2(np.fft.ifftshift(shift_g))[:,:,np.newaxis])
#     Gc = np.conj(G)
#     Gd_p = np.sqrt(G) # =Gd*np.conj(Gd)

#     F1 = np.fft.rfft2(img1, axes=(0,1))
#     F2 = np.fft.rfft2(img2, axes=(0,1))
#     F1c, F2c = np.conj(F1), np.conj(F2)
#     A1, A2 = F1*F1c, F2*F2c
#     Num = A1.shape[-1]
#     if Num==1:
#         nu = 0
#     PN = (np.sum(A1+A2, axis=-1, keepdims=True)-(A1+A2))/(Num-1+1e-8) # the mean of neg power 

#     nume = 2*(G+mu*Gd_p)*F1c*F2
#     deno = A1 + A2 + 2*lamda + 2*mu*Gd_p + 2*nu*PN*np.sqrt(G)
#     # R = nume/deno
#     R = (1+1.5*nu)* nume/deno # The (1+nu) is a constant, to make the max r close to 1.0
#     r = np.fft.irfft2(R, axes=(0,1)) 
#     shift_r = np.fft.fftshift(r, axes=(0,1))
#     return shift_r

if __name__ == '__main__':
    def preprocessing(img):
       img = img- np.mean(img)
       img = img/np.linalg.norm(img)
       return img

    import matplotlib.pyplot as plt
    from plot import plot_cc_map
        
    #- Visualize CC methods correlation map
    img1 = np.random.rand(32,32,1)
    img1 = preprocessing(img1)
    img2 = img1
     
    methods =['scc','pc', 'spof', 'rpc', 'sbcc']
    for method in methods:
        r = eval(method)(img1, img2)
        plot_cc_map(r[...,0], method.upper())
    plt.show()
