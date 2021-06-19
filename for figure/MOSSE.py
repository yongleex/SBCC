import cv2
import numpy as np

t = cv2.imread('target.png', 0)
cv2.imwrite('target_gray.png',t)

temp = cv2.imread('template.png', 0)
cv2.imwrite('template_gray.png', temp)

img1 = temp
x, y = np.meshgrid(np.arange(0,img1.shape[0],1), np.arange(0,img1.shape[0],1))
x, y = x-img1.shape[0]//2, y- img1.shape[1]//2
var = 2.25
shift_g = np.exp(-0.5*(x**2+y**2)/var)
G = np.fft.rfft2(np.fft.ifftshift(shift_g))

F1 = np.fft.rfft2(img1)

Ft = G*np.conj(F1)/(F1*np.conj(F1))
surrogate = np.fft.irfft2(Ft)

surrogate = surrogate - np.min(np.min(surrogate))
surrogate = surrogate / np.max(np.max(surrogate))*255
cv2.imwrite('xx.png', surrogate.astype(np.uint8))
import matplotlib.pyplot as plt

plt.imshow(surrogate)
plt.show()
