import numpy as np

def grid_window(image, step_sz=[16,16], win_sz=[32,32]):
    # image: W*H
    # wins:  w*h*B (interrogation windows)
    # posi:  2*B (center coords)

    assert len(image.shape)==2, "Only monocolor image is considered"
    W, H = image.shape
    w, h = win_sz[0], win_sz[1]
    sw, sh = step_sz[0], step_sz[1]

    # local index
    local_x, local_y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    local_x, local_y = local_x.reshape([w,h,1]), local_y.reshape([w,h,1])
    
    # start position of windows
    global_x, global_y = np.arange(0,W-w+1,sw), np.arange(0,H-h+1,sh)
    global_x, global_y = np.meshgrid(global_x, global_y, indexing="ij")
    shape = global_x.shape
    global_x, global_y = global_x.reshape([1,1,-1]), global_y.reshape([1,1,-1])

    # outputs
    wins = image[global_x+local_x, global_y+local_y]
    
    posix, posiy = global_x.reshape(-1)+w/2.0-0.5, global_y.reshape(-1)+h/2.0-0.5
    posi = np.stack([posix,posiy], axis=0)
    return wins, posi, shape

def unit_test():
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread("./img.png",0)
    if img is None:
        img = np.random.rand(512,512)
    
    wins, posi, shape = grid_window(img, step_sz=[75,75], win_sz=[150,150])

    plt.figure()
    plt.imshow(img, cmap="gray")

    plt.figure()
    for i in range(posi.shape[-1]):
        plt.subplot(shape[0],shape[1],i+1)
        plt.imshow(wins[:,:,i], cmap="gray")
        plt.axis("off")
        plt.axis("tight")
    plt.show()

if __name__ == "__main__":
    unit_test()
