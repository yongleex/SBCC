import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt


path1 = './3a.tif'
path2 = './3b.tif'

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

x = np.linspace(0,img1.shape[1], img1.shape[1])
y = np.linspace(0,img1.shape[0], img1.shape[0])
x,y = np.meshgrid(x,y)
# noisex = np.sin(2*np.pi*x/8) + 1
# noisey = np.sin(2*np.pi*y/8) + 1
noisex = (np.sin(2*np.pi*x/40)>0) + 1
noisey = (np.sin(2*np.pi*y/40)>0) + 1
noise = noisex*noisey

img1n = np.maximum(img1,  20*noise)
img2n = np.maximum(img2,  20*noise)

cv2.imwrite('./3a_n.tif', img1n.astype(np.uint8))
cv2.imwrite('./3b_n.tif', img2n.astype(np.uint8))

plt.imshow(noise)
plt.figure()
plt.imshow(img1n)
plt.show()


