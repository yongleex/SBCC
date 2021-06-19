import numpy as np
import cv2
from piv import PIV
from data import gen_image_pair
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from utils import tool
from tqdm import tqdm
import os


sns.set_style()
sns.set()

plt.style.use('seaborn-dark-palette')

def main():
    
    print("This exp may take serveral hours ...\n ")
    method_list  =['pc', 'spof', 'cfcc', 'scc', 'rpc', 'sbcc_b1', 'sbcc_b2', 'sbcc_b3', 'sbcc']
    line_styles = [(0,(1,1)), (0,(3,1)), (0,(5,1)), (0,(8,1)), (0,(1,1,2,2)),
            (0,(1,1,1,1,1,1,4,1)), (0,(1,1,1,1,4,1)), (0,(1,1,4,1)), (0,())]
    colors = cm.Dark2(np.linspace(0,1,9))
    line_styles = dict(zip(method_list, line_styles))
    colors = dict(zip(method_list, colors))


    disp_list = np.arange(0.0,10.01,0.1)  
    # disp_list = np.arange(0.0,2.01,0.1)  
    dp_list = [0.6, 2.2, 4.0]
    # dp_list = [0.6]
    runs = 100

    # config for piv 
    cfg_piv = tool.AttrDict()
    cfg_piv.win_sz = [32,32]
    cfg_piv.runs = 1
    cfg_piv.step_sz =[16,16]
    cfg_piv.subpixel='gaussian' # 'gaussian' # 'centroid' # 'parabolic'

    # config for PIG (particle image generator)
    cfg_pig = tool.AttrDict
    cfg_pig.img_sz = (256,256)
    cfg_pig.ppp = 0.02
    # cfg_pig.dp = 0.1  # [0.6,2.2, 4.0]
    cfg_pig.d_std = 0.01
    cfg_pig.i_std = 0.1
    cfg_pig.miss_ratio = 0.1
    
    for dp in dp_list:
        cfg_pig.dp = dp 

        for split in range(2):
            print(f"The {split}/2 splits (split the methods into 2 groups)")
            method_list = ['pc', 'spof', 'cfcc', 'scc', 'sbcc']if split==0 else ['rpc', 'sbcc_b1', 'sbcc_b2', 'sbcc_b3', 'sbcc']
            # Exp test
            result_rmse = np.zeros((len(disp_list),len(method_list), runs))
            result_bias = np.zeros((len(disp_list),len(method_list), runs))
            result_outlier = np.zeros((len(disp_list),len(method_list), runs))
            for run in range(runs):
                print(f"\tThe {run}/{runs} run...")
                for i, disp in tqdm(enumerate(disp_list)) :
                    cfg_pig.displacement = disp 
                    image1, image2= gen_image_pair(cfg_pig)
                    image1 = image1 + 5*np.random.rand(*image1.shape)
                    image2 = image2 + 5*np.random.rand(*image2.shape)
                    for j, method in enumerate(method_list):
                        cfg_piv.method = method
                        piv = PIV(cfg_piv)
                        u, v, r_map= piv.compute(image1, image2)

                        v_res, u_res = np.abs(v-disp), np.abs(u)
                        mask = (v_res>1) + (u>1) # masks
                        v_res[mask]=np.NaN
                        u_res[mask]=np.NaN
                        
                        rmse = np.nanmean(v_res**2)
                        bias = np.nanmean(v-disp)
                        outlier = np.sum(mask)/np.prod(mask.shape)
                        result_rmse[i,j,run]=rmse
                        result_bias[i,j,run]=bias
                        result_outlier[i,j,run]=outlier
                        info=f"disp:{disp:0.2f},method:{method:5s}, RMSE:{rmse:0.4f}, outlier:{outlier:5.0f}"
                        # print(info)

            # plot the results
            fig = plt.figure(figsize=(6,3))
            # plt.axes(yscale='log')
            for j, method in enumerate(method_list):
                plt.plot(disp_list, np.sqrt(np.mean(result_rmse[:,j,:],axis=-1)), linestyle=line_styles[method],color=colors[method], label=method.upper())
            plt.xlim(np.min(disp_list)-0.01, np.max(disp_list)+0.01)
            # plt.ylim(-1e-3, 0.3)
            plt.ylabel("RMSE")
            plt.xlabel("Displacement (Pixel)")
            plt.gcf().subplots_adjust(bottom=0.18)
            plt.legend(loc=1)
            plt.savefig(f'output/Fig8_rmse_{dp}_{split}.svg')
            plt.savefig(f'output/Fig8_rmse_{dp}_{split}.pdf')

            # fig = plt.figure(figsize=(6,3))
            # for j, method in enumerate(method_list):
            #     plt.plot(disp_list, np.mean(result_bias[:,j,:],axis=-1), linestyle=line_styles[method],color=colors[method], label=method.upper())
            # plt.xlim(np.min(disp_list)-0.01, np.max(disp_list)+0.01)
            # plt.ylabel("Bias")
            # plt.xlabel("Displacement (Pixel)")
            # plt.legend(loc=1)
            # plt.savefig(f'output/Fig8_bias_{dp}_{split}.svg')
            # plt.savefig(f'output/Fig8_bias_{dp}_{split}.pdf')

            # fig = plt.figure(figsize=(6,3))
            # for j, method in enumerate(method_list):
            #     plt.plot(disp_list, np.mean(result_outlier[:,j,:],axis=-1), linestyle=line_styles[method],color=colors[method], label=method.upper())
            # plt.xlim(np.min(disp_list)-0.01, np.max(disp_list)+0.01)
            # plt.ylabel("Outliers")
            # plt.xlabel("Displacement (Pixel)")
            # plt.legend(loc=1)
            # plt.savefig(f'output/Fig8_outler_{dp}_{split}.svg')
            # plt.savefig(f'output/Fig8_outler_{dp}_{split}.pdf')

    plt.show()
    
if __name__ == '__main__':
    main()
