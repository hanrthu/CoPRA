import sys
sys.path.append('/home/HR/PIXberts')
from data.complex import ComplexInput
from data.register import DataRegister
from torch.utils.data import Dataset
import pandas as pd
import esm
import torch
import yaml
from tqdm import tqdm
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
from tqdm import tqdm
import os
import math
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from data.structure_dataset import StructureDataset
from typing import Optional, Dict
from easydict import EasyDict
from data.structure_dataset import _process_structure
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter
import torch.nn.functional as F
from models.convs.modules import GaussianExpansion
from torch_geometric.data import Data

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()

@R.register('graph_dataset')
class GraphDataset(StructureDataset):
    ''' 
    The implementation of Protein-RNA structure Graph Dataset
    '''
    def __init__(self, 
                 dataframe, 
                 data_root, 
                 col_prot_name='PDB',
                 col_prot_chain='Protein chains',
                 col_na_chain='RNA chains',
                 col_prot='protein',
                 col_na='na',
                 col_label='△G(kcal/mol)',
                 diskcache=None,
                 transform=None,
                 **kwargs
                 ):
        super(GraphDataset, self).__init__(dataframe, 
                 data_root, 
                 col_prot_name,
                 col_prot_chain,
                 col_na_chain,
                 col_prot,
                 col_na,
                 col_label,
                 diskcache,
                 transform,
                 **kwargs)
        self.rbf = GaussianExpansion(max_value=10, K=16)
        
    def to_graph(self, data, atom=True):
        restype = data['restype']
        if not atom:
            pos = data['pos_atoms']
            mask = data['seq_mask']
            mean_pos = pos.mean(dim=-2)
            edge_index = knn_graph(x=mean_pos, k=32)
            row, col = edge_index
            rel_pos = mean_pos[row] - mean_pos[col]
            node_s = restype
        else:
            atom_pos = []
            atom_type = []
            pos = data['pos_heavyatom'] # L, 41, 3
            mask = data['mask_heavyatom'] # L, 41
            # print("Pos, Mask:", pos.shape, mask.shape)
            for i in range(len(pos)):
                pos_i = pos[i]
                mask_i = mask[i]
                pos_valid = pos_i[mask_i == 1]
                index_valid = torch.nonzero(mask_i).flatten()
                if len(index_valid) == 0:
                    continue
                # print("Index:", index_valid)
                atom_pos.append(pos_valid)
                atom_type.append(index_valid)
            pos = torch.cat(atom_pos, dim=0)
            atomtype = torch.cat(atom_type, dim=0)
            edge_index = knn_graph(x=pos, k=32)
            row, col = edge_index
            rel_pos = pos[row] - pos[col]
            node_s = atomtype

        d = rel_pos.norm(dim=-1)
        rel_pos = F.normalize(rel_pos, dim=-1)

        # GVP uses the distance expansion as scalar edge-feature of shape [Ne, self.edge_dims[0]]
        # and the normed relative position between two nodes as vector edge-feature shape [Ne, self.edge_dims[1],  3]

        # distance expansion
        d = self.rbf(d)
        
        data = Data(
            # seq = restype,                  # num_res x 1
            node_s = node_s,               # num_res x 1  
            node_pos = pos,
            edge_s = d,                     # num_edge x 1  
            edge_v = rel_pos.unsqueeze(-2), # num_edge x 1 x 3   
            edge_index = edge_index,        # 2 x num_edges
            # mask_coords = mask,             # num_res
            labels = data['labels']
        )
        # print(data)
        return data

    def __getitem__(self, idx):
        data = self.data[idx]
        # print("Before Transform:", data)
        if self.transform is not None:
            data = self.transform(data)
        data = self.to_graph(data)
        # print(data)
        return data