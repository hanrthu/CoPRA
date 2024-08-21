from torch.utils.data import Dataset
from data.register import DataRegister
import esm
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
import torch
from tqdm import tqdm
R = DataRegister()

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}


CLS_TOKEN_IDX = 0
PAD_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2

@R.register('sequence_dataset')
class SequenceDataset(Dataset):
    def __init__(self, dataframe,
                 col_prot='protein', col_na='na',col_label='dG', col_prot_name = 'PDB',
                 diskcache=None,
                 **kwargs):
        super(SequenceDataset, self).__init__()
        self.df = dataframe
        self.col_protein = col_prot
        self.col_prot_name = col_prot_name
        self.col_na = col_na
        self.col_label = col_label
        self.type = 'reg'
        self.diskcache = diskcache
        self.prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.na_alphabet = Alphabet(**na_alphabet_config)
        self.load_data()
    
    def load_data(self):
        self.data = []

        for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            structure_id = row[self.col_prot_name]
            if self.diskcache is None or structure_id not in self.diskcache:
                max_prot_length = 0
                max_na_length = 0
                prot_seqs_info = row[self.col_protein].split(',')
                na_seqs_info = row[self.col_na].split(',')
                prot_seqs = []
                na_seqs = []
                for prot_seq in prot_seqs_info:
                    if ':' in prot_seq:
                        prot_seq = prot_seq.split(':')[1]
                    if len(prot_seq) > max_prot_length:
                        max_prot_length = len(prot_seq)
                    prot_seqs.append(prot_seq)
                for na_seq in na_seqs_info:
                    if ':' in na_seq:
                        na_seq = na_seq.split(':')[1]
                    if len(na_seq) > max_na_length:
                        max_na_length = len(na_seq)
                    na_seqs.append(na_seq)
                    
                label = row[self.col_label]
                item = {
                    'id': structure_id,
                    'prot_seqs': prot_seqs,
                    'na_seqs': na_seqs,
                    'label': label,
                    'max_prot_length': max_prot_length,
                    'max_na_length': max_na_length
                }
                
                self.data.append(item)
                if self.diskcache is not None:
                    self.diskcache[structure_id] = item
            else:
                self.data.append(self.diskcache[structure_id])
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.df)

class CustomSeqCollate(object):
    def __init__(self, strategy='separate'):
        super(CustomSeqCollate, self).__init__()
        self.strategy=strategy
    def __call__(self, batch):
        size = len(batch)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float32)
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        # print("Batch:", batch)
        # for item in batch:
        #     for seq  in item['na_seqs']:
        #         print("RNA seqs:", seq, len(seq))
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['na_seqs']) for item in batch]
        if self.strategy == 'separate':
            max_item_prot_length = [item['max_prot_length'] for item in batch]
            max_item_na_length = [item['max_na_length'] for item in batch]
            max_prot_length = max(max_item_prot_length)
            max_na_length = max(max_item_na_length)
            total_prot_chains = sum(prot_chains)
            total_na_chains = sum(na_chains)
            prot_batch = torch.empty([total_prot_chains, max_prot_length+2])
            prot_batch.fill_(prot_alphabet.padding_idx)
            na_batch = torch.empty([total_na_chains, max_na_length+2])
            na_batch.fill_(na_alphabet.pad_idx)
            curr_prot_idx = 0
            curr_na_idx = 0
            for item in batch:
                prot_seqs = item['prot_seqs']
                na_seqs = item['na_seqs']
                # print(item['id'])
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

        elif self.strategy == 'combine':
            # prot_linker = 'G' * 25
            # na_linker = 'T' * 15
            prot_linker = ''
            na_linker = ''
            complex_prot_max_length = 0
            complex_na_max_length = 0
            for i, item in enumerate(batch):
                prot_seqs = item['prot_seqs']
                na_seqs = item['na_seqs']
                prot_complex_seq = prot_linker.join(prot_seqs)
                na_complex_seq = na_linker.join(na_seqs)
                if len(prot_complex_seq) > complex_prot_max_length:
                    complex_prot_max_length = len(prot_complex_seq)
                if len(na_complex_seq) > complex_na_max_length:
                    complex_na_max_length = len(na_complex_seq)
            
            prot_batch = torch.empty([len(batch), complex_prot_max_length+2])
            prot_batch.fill_(prot_alphabet.padding_idx)
            na_batch = torch.empty([len(batch), complex_na_max_length+5])
            na_batch.fill_(na_alphabet.pad_idx)
            
            for i, item in enumerate(batch):            
                prot_batch[i, 0] = prot_alphabet.cls_idx
                prot_complex_encode = prot_alphabet.encode(prot_complex_seq)
                seq = torch.tensor(prot_complex_encode, dtype=torch.int64)
                prot_batch[i, 1: len(prot_complex_encode)+1] = seq
                prot_batch[i, len(prot_complex_encode)+1] = prot_alphabet.eos_idx
                prot_linker_start = 1
                if len(item['prot_seqs']) > 1 and len(prot_linker) > 0:
                    # print("Combining Protein...", len(item['prot_seqs']))
                    for j, p_seq in enumerate(item['prot_seqs']):
                        if j == len(item['prot_seqs']) - 1:
                            break
                        seq_len = len(p_seq)
                        prot_linker_start += seq_len
                        linker_len = len(prot_linker)
                        prot_batch[i, prot_linker_start: prot_linker_start + linker_len] = prot_alphabet.padding_idx
                        # print("Done!", i)
                        prot_linker_start += linker_len
                # na_batch[i, 0] = na_alphabet.cls_idx
                na_complex_encode = na_alphabet.encode(na_complex_seq)
                seq = torch.tensor(na_complex_encode, dtype=torch.int64)
                na_batch[i, :len(seq)] = seq
                na_linker_start = 1
                if len(item['na_seqs']) > 1 and len(na_linker) > 0:
                    # print("Combining NA...", len(item['na_seqs']))
                    for j, n_seq in enumerate(item['na_seqs']):
                        if j == len(item['na_seqs']) - 1:
                            break
                        seq_len = len(n_seq)
                        na_linker_start += seq_len
                        linker_len = len(na_linker)
                        na_batch[i, na_linker_start: na_linker_start + linker_len] = na_alphabet.pad_idx
                        na_linker_start += linker_len
                # na_batch[i, len(na_complex_encode)+1] = na_alphabet.eos_idx
        else:
            raise ValueError
        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        
        data = {
            'size': size,
            'labels': labels,
            'prot': prot_batch.long(),
            'prot_chains': prot_chains,
            'protein_mask': prot_mask,
            'na': na_batch.long(),
            'na_chains': na_chains,
            'na_mask': na_mask, 
            'strategy': self.strategy
            }
        # print(data)
        return data