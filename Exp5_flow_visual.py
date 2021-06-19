import numpy as np
import cv2
from piv import PIV
import matplotlib.pyplot as plt
from utils import tool
from utils.outlier import NMT
from utils.plot import plot_cc_map
"""One-pass analysis for PIV with different CC methods"""


def plot_field(u, v, method, v_min=0.0, v_max=12.0):
    u =-u.transpose()
    v = v.transpose()
    amp = np.sqrt(u**2+v**2)
    fig = plt.figure()
    plt.imshow(amp, cmap='jet', interpolation='bicubic', vmin=v_min, vmax=v_max)
    ax = fig.gca()
    ax.quiver(u,v, scale=800/(v_max-7.5)) 
    ax.set_title(method)
    ax.axis('equal')
    ax.axis('off')
    plt.colorbar()
    return fig


def main():
    image_lists = [ ['./TestImages/2a.tif', './TestImages/2b.tif'],
            ['./TestImages/F_00001.bmp', './TestImages/F_00002.bmp'],
            ['./TestImages/5a_piv01_1.bmp', './TestImages/5b_piv01_2.bmp'],
            # ['./TestImages/A001_1.tif', './TestImages/A001_2.tif'],
            # ['./TestImages/3a.tif', './TestImages/3b.tif'],
            # ['./TestImages/4a.tif', './TestImages/4b.tif'],
            ]
    v_max_list = [10, 9, 12, 12, 10] # for visualization control, max value for the vector amplitude 

    method_list  =['pc', 'spof', 'cfcc', 'scc', 'rpc', 'sbcc_b1', 'sbcc_b2', 'sbcc_b3', 'sbcc']
    
    # config for piv 
    config = tool.AttrDict()
    config.win_sz = [32,32]
    config.step_sz =[16,16]
    config.subpixel='gaussian'
    # config.runs= 1 # One-pass method, you also can test the multi-pass method with image deformation

    for k, (file_path, v_max) in enumerate(zip(image_lists, v_max_list)):
        image1 = cv2.imread(file_path[0], 0)
        image2 = cv2.imread(file_path[1], 0)

        # get the "truth" reference for outlier identification
        config.runs = 3
        config.method = 'scc'
        piv = PIV(config)
        u_t, v_t, r_map= piv.compute(image1, image2)
        u_t, v_t, index = NMT(u_t,v_t) # save the data to generate truth, please make config.runs=3

        for method in method_list:
            info = f"{v_max}_{method}"

            config.method = method
            config.runs = 1
            piv = PIV(config)

            u, v, r_map= piv.compute(image1, image2)
            
            cri = np.sqrt(np.square(u-u_t) + np.square(v-v_t)) > 4
            print(f"{info:10s}\toutlier number:\t {np.sum(cri):4d}")

            fig = plot_field(u,v,method.upper(), v_max=v_max)
            fig.savefig('output/Figx_'+info+".svg")
            # plt.close()
        print('\n')

    plt.show()

if __name__ == '__main__':
    main()
