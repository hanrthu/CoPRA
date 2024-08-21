import random
import torch

from ._base import register_transform

@register_transform('subtract_center_of_mass')
class SubtractCOM(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        pos = data['pos_atoms']
        mask = data['mask_atoms']
        if mask is None:
            center = np.zeros(3)
        elif mask.sum() == 0:
            center = np.zeros(3)
        else:
            center = pos[mask].mean(axis=0)
        data['pos_atoms'] = pos - center[None, None, :]
        return data
    
