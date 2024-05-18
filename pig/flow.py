import numpy as np
import cv2
import matplotlib.pyplot as plt


def uniform_flow(x, y, c_x=10, c_y=0):
    """
    uniform flow with a constant velocity value
    v(x,y)=c
    """
    u = np.zeros_like(x)+c_x
    v = np.zeros_like(x)+c_y
    return u, v


def sine_flow(x, y, a=6, px=128, vmax=20):
    """
    sine flow along a sinusoidal stream line, |v|=vmax
    u(x) = vmax*\frac{1}{\sqrt(1+(2\pi a cos (2\pi x/px)/b)^2)}
    v(x) = vmax*\frac{2\pi a cos (2\pi x/px)/b}{\sqrt(1+(2\pi a cos (2\pi x/px)/b)^2)}
    """
    temp = 2*np.pi*a*np.cos(2*np.pi*x/px)/px
    u = vmax/np.sqrt(1+temp**2)
    v = vmax*temp/np.sqrt(1+temp**2)
    return u, v


def lamboseen_flow(x, y, gamma=2e3, rc=40, x_c=None, y_c=None):
    """
    Lamb-Oseen vortex as detailed
    https://doi.org/10.1109/TIM.2021.3132999
    """
    if x_c is None or y_c is None:
        x_c, y_c = x.shape[0]//2, x.shape[1]//2
    x, y = x-x_c, y-y_c
    r = np.sqrt(x**2+y**2)+1e-15
    theta = np.arctan2(y,x)

    d_s= gamma*(1-np.exp(-r**2/rc**2))/(2*np.pi*r)  # circumferential vel
    d_theta = d_s/r
    u = -d_s*np.sin(theta)
    v = d_s*np.cos(theta)
    return u, v


# def lamboseen_flow2(x, y, gamma=2e3, rc=40, x_c=None, y_c=None):
#     """
#     Lamb-Oseen vortex as detailed
#     https://doi.org/10.1109/TIM.2021.3132999
#     """
#     if x_c is None or y_c is None:
#         x_c, y_c = x.shape[0]//2, x.shape[1]//2
#     x, y = x-x_c, y-y_c
#     r = np.sqrt(x**2+y**2)+1e-15
#     theta = np.arctan2(y,x)

#     d_s= gamma*(1-np.exp(-r**2/rc**2))/(2*np.pi*r)  # circumferential vel
#     d_theta = d_s/r

#     dx = r*np.cos(theta+d_theta)-x # The next positions
#     dy = r*np.sin(theta+d_theta)-y
#     return dx, dy


def cellular_flow(x, y, vmax=10, px=128, py=128):
    """

    """
    u =  vmax*np.sin(2*np.pi*x/px)*np.cos(2*np.pi*y/py)
    v = -vmax*np.cos(2*np.pi*x/px)*np.sin(2*np.pi*y/py)
    return u, v


def displacement(flow, x, y, n=512):
    """
    Runge-Kutta method to evaluate the next position (x_n, y_n)
    \frac{dx}{dt} = v(x)
    The error is less than 1e-9, that works fine for this work
    """
    xn, yn = x,y
    h = 1./n
    for _ in range(n):
        k1_x, k1_y = flow(xn,yn)
        k2_x, k2_y = flow(xn+k1_x*h/2, yn+k1_y*h/2)
        k3_x, k3_y = flow(xn+k2_x*h/2, yn+k2_y*h/2)
        k4_x, k4_y = flow(xn+k3_x*h, yn+k3_y*h)
        xn = xn + h*(k1_x+2*k2_x+2*k3_x+k4_x)/6
        yn = yn + h*(k1_y+2*k2_y+2*k3_y+k4_y)/6
    return xn, yn


flow_dict={"uniform_flow":uniform_flow, "sine_flow":sine_flow, 
           "lamboseen_flow":lamboseen_flow, "cellular_flow":cellular_flow}


def test():
    from plot import plot_field

    x,y = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")

    for flow in [uniform_flow, sine_flow, lamboseen_flow, cellular_flow]:
        u, v = flow(x,y)
        x_n, y_n = displacement(flow, x, y)
        dx, dy = x_n-x, y_n-y
            
        delta = np.sqrt((u-dx)**2+(v-dy)**2)
        # fig = plot_field(x,y,u,v); 
        fig = plot_field(x,y,u,v,delta)
        plt.title(f"The synthetic flow field: {flow.__name__}")
    
    plt.show()


if __name__ == "__main__":
    test()
