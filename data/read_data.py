import numpy as np


path = './data/uniform/uniform_00001_flow.flo'

def read_flow(path):
    TAG_FLOAT = 202021.25
    with open(path) as f:
        raw = np.fromfile(f, np.float32)
        tag, width, height = raw[0:3]
        if tag !=TAG_FLOAT:
            raise ValueError(f"The check code for file({path}) is not correct")
        width = int.from_bytes(width.tobytes(),'little')
        height = int.from_bytes(height.tobytes(),'little')
        data = raw[3:]
        data = np.reshape(data, (width,height,2))
    return data

data = read_flow(path)
print(data[:,:,0])
print(data[:,:,1])
