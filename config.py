import numpy as np


class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    # one-pass cross correlation configuration
    cc = AttrDict()
    cc.step_sz=[[16,16],[8,8],[4,4]] # for Multi-Pass cross correlation
    cc.win_sz =[[32,32],[16,16],[16,16]]
    cc.pre_norm=True          # image pre-processing for interrogation wins
    cc.cc_method = "rpc"      # ['scc','pc', 'spof', 'rpc', 'sbcc']
    cc.sub_method= "gaussian" # ["centroid", "parabolic", "gaussian"]

    cfg = AttrDict()
    cfg.cc = cc
    return cfg

