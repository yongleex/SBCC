import numpy as np
import cv2


def grid_window(image, step_sz=[16,16], win_sz=[32,32]):
    # start position of windows
    start_x = np.arange(0,(image.shape[0]-win_sz[0]+1),step_sz[0])
    # print(start_x)
    start_y = np.arange(0,(image.shape[1]-win_sz[1]+1),step_sz[1])
    start_x, start_y = np.meshgrid(start_x, start_y)
    vec_shape = start_x.shape
    start_x, start_y = np.reshape(start_x, (1,1,-1)), np.reshape(start_y, (1,1,-1))


    # all the windows index 
    win_x, win_y = np.meshgrid(np.arange(0, win_sz[0]), np.arange(0, win_sz[1]))
    win_x = win_x[:,:, np.newaxis] + start_x
    win_y = win_y[:,:, np.newaxis] + start_y

    windows = image[win_x, win_y]
    return windows, vec_shape


def main():
    """test the window technology"""
    import matplotlib.pyplot as plt

    image = np.random.randn(512, 512)
    image = np.random.randn(64,64)
    # image = cv2.imread('./TestImages/16.tif',0)[:128,:64]
    step_sz = [16,16]
    win_sz =[32,32]

    windows,_ = grid_window(image, step_sz, win_sz)
    plt.imshow(image[:,:])

    for i in range(4):
        plt.figure()
        plt.imshow(windows[:,:,i].transpose())

    print(windows.shape)
    plt.show()


if __name__ =='__main__':
    main()
