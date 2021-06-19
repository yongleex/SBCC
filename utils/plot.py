import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
sns.set_style()


def plot_cc_map(r, title="test"):
    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.meshgrid(np.arange(0,r.shape[0],1), np.arange(0,r.shape[0],1))
    x, y = x-r.shape[0]//2, y- r.shape[1]//2
    ax.plot_surface(x[:],y[:],r[:], cmap='rainbow')
    ax.set_zlim(-0.1,1.0)
    # ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    # ax.axis('off')
    return fig
