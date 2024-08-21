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
from typing import Optional, Dict
from easydict import EasyDict
from data.protein.residue_constants import restype_order, restype_num
from data.rna.base_constants import RNA_NUCLEOTIDES

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()
# ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
# ATOM_P, ATOM_C4, ATOM_NB = 37, 38, 

def _process_structure(structure_path, structure_id, valid_prot_chains=None, valid_rna_chains=None, gpu=None) -> Optional[Dict]:
    cplx = ComplexInput.from_path(structure_path, valid_prot_chains=valid_prot_chains, valid_rna_chains=valid_rna_chains)
    if cplx is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {structure_path}')
        return None

    data = EasyDict({
        'seq': cplx.seq, 'prot_seqs': cplx.prot_seqs, 'rna_seqs': cplx.na_seqs, 'res_nb': torch.LongTensor(cplx.res_nb),
        'chain_nb': torch.LongTensor(cplx.chainid), 'identifier': torch.LongTensor(cplx.identifier),
        'restype': torch.LongTensor(cplx.restype), 'seq_mask': torch.BoolTensor(cplx.mask),
        'pos_heavyatom': torch.FloatTensor(cplx.atom41_positions), 'mask_heavyatom': torch.BoolTensor(cplx.atom41_mask),
        'atom64_positions': torch.FloatTensor(cplx.atom_positions), 'atom64_mask': torch.BoolTensor(cplx.atom_mask), 
    })
    if gpu is not None:
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(gpu)
    data['id'] = structure_id
    return data


@R.register('structure_dataset')
class StructureDataset(Dataset):
    ''' 
    The implementation of Protein-RNA structure Dataset
    '''
    def __init__(self, 
                 dataframe, 
                 data_root, 
                 col_prot_name='PDB',
                 col_prot_chain='Protein chains',
                 col_na_chain='RNA chains',
                 col_prot='Protein sequences',
                 col_na='RNA sequences',
                 col_label='△G(kcal/mol)',
                 diskcache=None,
                 transform=None,
                 mut=False,
                 col_mut='Mutation sequences',
                 **kwargs
                 ):
        self.data_root = data_root
        self.df: pd.DataFrame = dataframe.copy()
        self.col_prot_name = col_prot_name
        self.col_prot_chain = col_prot_chain
        self.col_na_chain = col_na_chain
        self.col_label = col_label
        self.col_prot = col_prot
        self.col_na = col_na
        self.type = 'reg'
        self.diskcache = diskcache
        self.prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.na_alphabet = Alphabet(**na_alphabet_config)
        self.mut = mut
        self.col_mut = col_mut
        
        self.transform = get_transform(transform)
        
        self.load_data()
        
    def load_data(self):
        self.data = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            structure_id = row[self.col_prot_name]
            complex = row[self.col_prot_name]
            if self.mut:
                structure_id += '_' + row['MUTATION']
            if self.diskcache is None or structure_id not in self.diskcache:
                print("Processing:", structure_id)
                prot_chains = row[self.col_prot_chain].split(',')
 
                na_chains = row[self.col_na_chain].split(',')
                if self.mut:
                    pdb_path = os.path.join(self.data_root, structure_id.split('_')[0]+'.pdb')
                else:
                    pdb_path = os.path.join(self.data_root, structure_id+'.pdb')

                label = row[self.col_label]
                
                cplx = _process_structure(pdb_path, structure_id, prot_chains, na_chains)

                if self.mut:
                    prot_mut = row[self.col_mut]
                    mut_list = prot_mut.split(',')
                    mut_list_to_type = []
                    for mut_seq in mut_list:
                        mut_seq = mut_seq[2:]
                        for res in mut_seq:
                            restype_idx = restype_order.get(res, restype_num)
                            mut_list_to_type.append(restype_idx)
                    na = row[self.col_na]
                    na_list = na.split(',')
                    for na_seq in na_list:
                        na_seq = na_seq[2:]
                        for na in na_seq:
                            if na in RNA_NUCLEOTIDES:
                                na_idx = RNA_NUCLEOTIDES.index(na) + 21
                            else:
                                na_idx = len(RNA_NUCLEOTIDES) + 21
                            mut_list_to_type.append(na_idx)
                    mut_seqs = [i[2:] for i in mut_list]
                    mut_restype = torch.tensor(mut_list_to_type, device=cplx['restype'].device)
                    mut_identifier = mut_restype != cplx['restype']
                    assert len(mut_restype) == len(cplx['restype'])
                    if (mut_restype != cplx['restype']).sum().item() != 1:
                        print('Name:', structure_id)
                        print("Mut:", mut_restype)
                        print("Wild:", cplx['restype'])
                        print("Diff:", (mut_restype != cplx['restype']).sum())
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
                if self.mut:
                    item = {
                        'complex': complex,
                        'labels': label,
                        'atom_min_dist': atom_min_dist, # needs 2D padding
                        'max_prot_length': max_prot_length,
                        'max_na_length': max_na_length,
                        'mut_seqs': mut_seqs,
                        'mut_restype': mut_restype,
                        'mut_identifier': mut_identifier
                }
                else:
                    item = {
                        'complex': complex,
                        'labels': label,
                        'atom_min_dist': atom_min_dist, # needs 2D padding
                        'max_prot_length': max_prot_length,
                        'max_na_length': max_na_length
                    }
                    
                cplx.update(item)
                # print("Complex {} is:".format(i), cplx)
                self.data.append(cplx)
                if self.diskcache is not None:
                    self.diskcache[structure_id] = cplx
            
            else:
                data = self.diskcache[structure_id]
                data['complex'] = complex
                self.data.append(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        # print("Before Transform:", data)
        if self.transform is not None:
            data = self.transform(data)
        # for key in data:
        #     if isinstance(data[key], torch.Tensor) or isinstance(data[key], str):
        #         data[key] = data[key][:256]
        # print("After Transform:", data)
        return data

EXCLUDE_KEYS = ['labels', 'complex']
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mask_atoms': 0,
    'chain_nb': -1,
}

class CustomStructCollate(object):
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

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_not_pad = []
        keys_ignore = ['prot_seqs', 'rna_seqs', 'mut_seqs', 'max_prot_length', 'max_na_length', 'atom_min_dist']
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
            for k in keys_not_pad:
                data_padded[k] = data[k]
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_berts(self, strategy, batch):
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        mut_flag = 0
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
        if 'mut_seqs' in batch[0]:
            mut_flag = 1
            mut_batch = torch.empty([total_prot_chains, max_prot_length])
            mut_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            if 'mut_seqs' in item:
                mut_seqs = item['mut_seqs']
            na_seqs = item['rna_seqs']
            for i, prot_seq in enumerate(prot_seqs):
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                if 'mut_seqs' in item:
                    mut_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                    mut_seq_encode = prot_alphabet.encode(mut_seqs[i])
                    seq_m = torch.tensor(mut_seq_encode, dtype=torch.int64)
                    mut_batch[curr_prot_idx, 1: len(mut_seq_encode)+1] = seq_m
                    mut_batch[curr_prot_idx, len(mut_seq_encode)+1] = prot_alphabet.eos_idx
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
        if mut_flag:
            return prot_batch.long(), mut_batch.long(), prot_chains, prot_mask, na_batch.long(), na_chains, na_mask
        else:
            return prot_batch.long(), prot_chains, prot_mask, na_batch.long(), na_chains, na_mask

    def __call__(self, data_list):
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        if 'mut_seqs' in data_list[0]:
            prot_batch, mut_batch, prot_chains, prot_mask, na_batch, na_chains, na_mask = self.pad_for_berts(self.strategy, data_list)
            batch['prot_mut'] = mut_batch
        else:
            prot_batch, prot_chains, prot_mask, na_batch, na_chains, na_mask = self.pad_for_berts(self.strategy, data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        batch['strategy'] = self.strategy
        batch['labels'] = batch['labels'].float()
        # print("This is my new batch!")
        # for key, value in batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(key, value.shape)
        #     else:
        #         print(key, value)
        return batch
    
if __name__ == '__main__':
    # print(os.path.exists('/home/HR/PIXberts/datasets/PNA350/splits/merge_filtered_demo.csv'))
    df = pd.read_csv('/home/HR/PIXberts/datasets/PNA350/splits/merge_filtered_demo.csv')
    df_train = df[df['fold_0'].isin(['train'])]
    
    with open('/home/HR/PIXberts/config/datasets/pdbbind_struct.yml', 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
    train_dataset = StructureDataset(df_train, **config_dict, diskcache=None)
    a = train_dataset[0]
    
    