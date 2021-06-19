import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_cc_map(r, title):
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(np.arange(0,r.shape[0],1), np.arange(0,r.shape[0],1))
    x, y = x-r.shape[0]//2, y- r.shape[1]//2
    ax.plot_surface(x[:],y[:],r[:], cmap='rainbow')
    ax.set_zlim(-0.1,1.0)
    ax.set_xlim(-16,16)
    ax.set_ylim(-16,16)
    # ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

x, y = np.meshgrid(np.arange(0,32,1), np.arange(0,32,1))
x, y = x-16, y-16

r = np.exp(-(x**2+y**2)*0.5/2.25)

plot_cc_map(r, "xx")

plt.savefig("xx.pdf")
plt.close()
plot_cc_map(r*0, "xx")
plt.savefig("yy.pdf")
