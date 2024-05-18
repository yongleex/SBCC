# default configure
class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    # configuration for PIG
    cfg = AttrDict()
    cfg.ppp = 0.05
    cfg.img_sz = [512,512]
    cfg.img_bd = 100

    # diameters
    cfg.sigma_d = 0.2
    cfg.mu_d = 2.5

    # particle intensity
    cfg.sigma_i = 0.1
    cfg.mu_i = 0.85

    # flow settings
    cfg.flow = "sine_flow" # "uniform_flow", "sine_flow", "lamboseen_flow", "cellular_flow"
    cfg.uniform_flow   ={'c_x':10., 'c_y':0.} # uniform_flow
    cfg.sine_flow      ={'a':5, 'px':128, 'vmax':5} # uniform_flow
    cfg.lamboseen_flow ={'gamma':2e3, 'rc':50, 'x_c':256, 'y_c':256} # uniform_flow
    cfg.cellular_flow  ={'vmax':10, 'px':128, 'py':128} # uniform_flow
    
    return cfg