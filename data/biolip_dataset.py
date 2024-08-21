import sys
sys.path.append('/home/HR/PIXberts')
from data.complex import ComplexInput
from data.register import DataRegister
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import esm
import torch
import yaml
from tqdm import tqdm
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
from tqdm import tqdm
import diskcache
import os
import math
import time
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from typing import Optional, Dict
from easydict import EasyDict
from data.structure_dataset import _process_structure

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()
# ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
# ATOM_P, ATOM_C4, ATOM_NB = 37, 38, 
@R.register('biolip_dataset')
class BioLiPDataset(Dataset):
    ''' 
    The implementation of Protein-RNA structure Dataset
    '''
    def __init__(self, 
                 dataframe, 
                 data_root, 
                 col_prot_name='PDB',
                 col_prot_chain='Protein chains',
                 col_na_chain='RNA chains',
                 col_binding_site='Binding site renumbered merged',
                 col_ligand='Binding ligands',
                 diskcache=None,
                 transform=None,
                 **kwargs
                 ):
        self.data_root = data_root
        self.df: pd.DataFrame = dataframe.copy()
        self.df.reset_index(drop=True, inplace=True)
        self.col_prot_name = col_prot_name
        self.col_prot_chain = col_prot_chain
        self.col_na_chain = col_na_chain
        self.col_binding_site = col_binding_site
        self.col_ligand = col_ligand
        self.diskcache = diskcache
        self.prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.na_alphabet = Alphabet(**na_alphabet_config)
        
        self.transform = get_transform(transform)
        
        # self.load_data()
        
    def load_data(self, idx):
        row = self.df.loc[idx]
        structure_id = row[self.col_prot_name]
        prot_chains = [row[self.col_prot_chain]]
        na_chains = [row[self.col_na_chain]]
        structure_id = structure_id + '_' + prot_chains[0] + '_' + na_chains[0]
        if self.diskcache is None or structure_id not in self.diskcache:
            # print("3UZS!")
            #  or structure_id in ['3UZS', '5W6V']
            structure_name = structure_id + '.cif'
            pdb_path = os.path.join(self.data_root, structure_name)
            
            cplx = _process_structure(pdb_path, structure_id, prot_chains, na_chains, gpu='cuda:5')
            
            # binding_sites = row[self.col_binding_site].split(',')
            # label_site = torch.zeros(len(cplx['seq'])).long()
            # for site in binding_sites:
            #     res = site[0]
            #     num = int(site[1:])
            #     # assert cplx['seq'][num-1] == res
            #     label_site[num-1] = 1
            
            ligand_id = row[self.col_prot_name] + '_' + row[self.col_na_chain]
            L = len(cplx['seq'])
            gpu_atoms = cplx['pos_heavyatom']
            gpu_masks = cplx['mask_heavyatom']
            distance_map = torch.linalg.norm(gpu_atoms[:, None, :, None, :]- gpu_atoms[None, :, None, :, :], dim=-1, ord=2).reshape(L, L, -1)
            mask = (gpu_masks[:, None, :, None] * gpu_masks[None, :, None, :]).reshape(L, L, -1)
            distance_map[~mask] = torch.inf
            atom_min_dist = torch.min(distance_map, dim=-1)[0]
            
            max_prot_length = 0
            max_na_length = 0
            for prot_seq in cplx.prot_seqs:
                if len(prot_seq) > max_prot_length:
                    max_prot_length = len(prot_seq)
            for na_seq in cplx.rna_seqs:
                if len(na_seq) > max_na_length:
                    max_na_length = len(na_seq)
    
            item = {
                'ligand_id': ligand_id, # no need to pad
                'atom_min_dist': atom_min_dist, # needs 2D padding
                # 'label_site': label_site, # needs padding 
                'max_prot_length': max_prot_length, # will be ignored in batching
                'max_na_length': max_na_length, # will be ignored in batching
                'can_bind': row[self.col_ligand] # no need to pad
            }
            
            cplx.update(item)
            # print("Complex {} is:".format(i), cplx)
            if self.diskcache is not None:
                # print("Writing!")
                for key in cplx:
                    if isinstance(cplx[key], torch.Tensor):
                        cplx[key] = cplx[key].detach().cpu()
                self.diskcache[structure_id] = cplx
            return cplx
            # break
        else:
            return self.diskcache[structure_id]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.load_data(idx)
        # print("Before Transform:", data)
        # gpu = 'cuda:4'
        # if gpu is not None:
        #     for key in data:
        #         if isinstance(data[key], torch.Tensor):
        #             data[key] = data[key].to(gpu)
        if self.transform is not None:
            data = self.transform(data)
        return data

EXCLUDE_KEYS = []
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mask_atoms': 0,
    'chain_nb': -1,
    'identifier': -1
}

class BioLiPStructCollate(object):
    def __init__(self, strategy='separate', length_ref_key='restype', pad_values=DEFAULT_PAD_VALUES, exclude_keys=EXCLUDE_KEYS, eight=True):
        super().__init__()
        self.strategy = strategy
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.exclude_keys = exclude_keys
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def pad_2d(self, x, n, value=10000):
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] == x.shape[1]
        if x.size(0) == n and x.size(1) == n:
            return x
        pad_size_1 = [n - x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size_1, fill_value=value).to(x)
        x_padded_1 = torch.cat([x, pad], dim=0)
        pad_size_2 = [x_padded_1.shape[0]] + [n - x_padded_1.size(1)] + list(x_padded_1.shape[2:])
        pad = torch.full(pad_size_2, fill_value=value).to(x)
        x_padded = torch.cat([x_padded_1, pad], dim=1)
        return x_padded
            
        

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_not_pad = []
        keys_ignore = ['prot_seqs', 'rna_seqs', 'max_prot_length', 'max_na_length', 'can_bind', 'ligand_id', 'atom_min_dist']
        pad_2d = ['atom_min_dist']
        for key in keys_inter:
            if key in keys_ignore:
                continue
            elif key not in self.exclude_keys:
                keys.append(key)
            else:
                keys_not_pad.append(key)
    
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            # print("Keys:", keys)
            # print("Datapadded Keys:", data_padded.keys())
            for k in pad_2d:
                data_padded[k] = self.pad_2d(data[k], max_length)
                
            for k in keys_not_pad:
                data_padded[k] = data[k]
                
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_berts(self, batch):
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        # print("Batch:", batch)
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]
        max_item_prot_length = [item['max_prot_length'] for item in batch]
        max_item_na_length = [item['max_na_length'] for item in batch]
        max_prot_length = max(max_item_prot_length)
        max_na_length = max(max_item_na_length)
        total_prot_chains = sum(prot_chains)
        total_na_chains = sum(na_chains)
        if self.eight:
            max_prot_length = math.ceil((max_prot_length + 2) / 8) * 8
            max_na_length =  math.ceil((max_na_length + 2) / 8) * 8
        else:
            max_prot_length = max_prot_length + 2
            max_na_length = max_na_length + 2
        prot_batch = torch.empty([total_prot_chains, max_prot_length])
        prot_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            na_seqs = item['rna_seqs']
            for prot_seq in prot_seqs:
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                curr_prot_idx += 1
            for na_seq in na_seqs:
                # na_batch[curr_na_idx, 0] = na_alphabet.cls_idx
                # NA encoder adds CLS and EOS by default
                na_seq_encode = na_alphabet.encode(na_seq)
                seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                na_batch[curr_na_idx, :len(seq)] = seq
                # na_batch[curr_na_idx, len(na_seq_encode)+1] = na_alphabet.eos_idx
                curr_na_idx += 1

        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        return prot_batch.long(), prot_chains, prot_mask, na_batch.long(), na_chains, na_mask

    def gen_clip_label(self, ligand_ids, can_bind_info):
        clip_label = torch.eye(len(ligand_ids))
        for i, can_bind in enumerate(can_bind_info):
            candidates = can_bind.split(',')
            for j, ligand_id in enumerate(ligand_ids):
                if ligand_id in candidates:
                    clip_label[i, j] = 1
        return clip_label.long()
                    

    def __call__(self, data_list):
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        prot_batch, prot_chains, prot_mask, na_batch, na_chains, na_mask = self.pad_for_berts(data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        batch['strategy'] = self.strategy
        ligand_ids =  [item['ligand_id'] for item in data_list]
        can_bind_info = [item['can_bind'] for item in data_list]
        batch['clip_label'] = self.gen_clip_label(ligand_ids, can_bind_info)
        # for key in batch:
        #     if isinstance(batch[key], torch.Tensor):
        #         batch[key] = batch[key].detach().cpu()
        # batch['labels'] = batch['labels'].float()
        return batch
    
if __name__ == '__main__':
    # print(os.path.exists('/home/HR/PIXberts/datasets/PNA350/splits/merge_filtered_demo.csv'))
    df = pd.read_csv('/home/HR/PIXberts/datasets/BioLiP2/pretrain_length_750_clean.csv')
    df_train = df
    
    with open('/home/HR/PIXberts/config/datasets/biolip.yml', 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
    cache = diskcache.Cache(directory='/home/HR/PIXberts/cache/biolip', eviction_policy='none')
    train_dataset = BioLiPDataset(df_train, **config_dict, diskcache=cache)
    # collate = BioLiPStructCollate
    start = time.time()
    # loader = DataLoader(
    #         train_dataset,
    #         batch_size=32,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=False,
    #         persistent_workers=False,
    #         collate_fn=collate(),)
    a = 0
    for i in tqdm(range(len(train_dataset))):
        _ = train_dataset[i]
        a += 1
    end = time.time()
    print("Total numbers:", a)
    print("Total Time Cost:", end-start)
    # print(a)
    