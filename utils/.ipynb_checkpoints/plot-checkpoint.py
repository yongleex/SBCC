import numpy as np
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D


def plot_field(x,y,u,v,bkg=None,cmap=None,figsize=(8,6)):
    assert len(x.shape) == 2, "the 2D data is required"
    def auto_step(x):
        sz = x.shape
        dx,dy=sz[0]//33+1, sz[1]//33+1
        return dx,dy
    
    fig=plt.figure(figsize=figsize)
    if bkg is not None:
        plt.imshow(bkg, cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(x*0+1,cmap="gray",vmax=1.0,vmin=0.0)

    dx,dy = auto_step(x)
    plt.quiver(y[::dx, ::dy], x[::dx, ::dy], v[::dx, ::dy], -u[::dx, ::dy])
    plt.axis('off')
    return fig
    
    
def plot_correlation_map(r, cmap='jet',figsize=None):
    r_sz = r.shape
    x, y = np.meshgrid(np.arange(r_sz[0]), np.arange(r_sz[1]),indexing="ij")
    x, y = x-r_sz[0]//2, y- r_sz[1]//2

    fig = plt.figure(figsize=figsize)
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(x[:],y[:],r[:], cmap=cmap)
    ax.set_zticklabels([])
    return fig

def test():
    sz = [128,256]
    v_max, lambx, lamby=25, 128, 128

    x,y = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing="ij")
    u =  v_max*np.sin(2*np.pi*x/lambx)*np.cos(2*np.pi*y/lamby)
    v = -v_max*np.cos(2*np.pi*x/lambx)*np.sin(2*np.pi*y/lamby)
    amp = np.sqrt(u**2+v**2)

    fig = plot_field(x,y,u,v)
    plt.savefig("temp1.pdf")
    fig = plot_field(x,y,u,v,bkg=amp,cmap=None,figsize=(8,3.2))
    plt.savefig("temp2.pdf")

    test_correlation_data  = np.random.rand(32,32)
    fig = plot_correlation_map(test_correlation_data)
    plt.show()

if __name__ == "__main__":
    test() 
    pass
