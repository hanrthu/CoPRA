from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, SparseTensor

def singleton(cls):
    _instance = {}

    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


class MyData(Data):
    chain: pd.Series
    channel_weights: torch.Tensor

    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None,
                edge_attr: OptTensor = None, y: OptTensor = None,
                pos: OptTensor = None, **kwargs
    ):
        super(MyData, self).__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        elif 'interaction' in key:
            return self.num_edges
        elif 'chains' in key:
            return self.num_chains
        else:
            return 0
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key or 'interaction' in key:
            return -1
        else:
            return 0
    @property
    def num_chains(self):
        return len(torch.unique(self.chains))