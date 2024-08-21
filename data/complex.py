import numpy as np
import dataclasses
import sys
from pathlib import Path
import io
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass
sys.path.append('/home/HR/PIXberts/')
import data.protein.proteins as proteins
from data.protein.atom_convert import atom37_to_atom14
from data.protein.proteins import chains_from_cif_string, chains_from_pdb_string
import data.rna.rnas as rnas

rna_residues = ['A', 'G', 'C', 'U']
protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
SUPER_PROT_IDX = 27
SUPER_RNA_IDX = 28
SUPER_CPLX_IDX = 29
SUPER_CHAIN_IDX = 4
PADDING_NODE_IDX = 26

@dataclass
class ComplexInput:
    seq: str # L
    mask: np.ndarray # (L, )
    restype: np.ndarray # (L, ) # In total 21 + 5 = 26 types, including '_' and 'X'
    res_nb: np.ndarray # (L, )
    prot_seqs: list
    na_seqs: list
    atom_mask: np.ndarray # (L, 37 + 27)
    atom_positions: np.ndarray #(L, 37 + 27, 3)
    
    atom41_mask: np.ndarray # (L, 14 + 27)
    atom41_positions: np.ndarray #(L, 14 + 27, 3)
    
    identifier: np.ndarray #(L, ), to identify rna or protein
    chainid: np.ndarray # (L, ), to identify chain of the complex
    
    @classmethod
    def from_path(self, path, valid_rna_chains=None, valid_prot_chains=None):
        if isinstance(path, io.IOBase):
            file_string = path.read()
        else:
            # print(path)
            path = Path(path)
            file_string = path.read_text()
        if valid_prot_chains is None or valid_rna_chains is None:
            valid_prot_chains = []
            valid_rna_chains = []
            if '.pdb' in str(path):
                chains = chains_from_pdb_string(file_string)
            elif '.cif' in str(path):
                chains = chains_from_cif_string(file_string)
                
            for chain in chains:
                for residue in chain:
                    if residue.get_resname() in protein_residues: 
                        valid_prot_chains.append(chain.get_full_id()[2])
                        break
                    if residue.get_resname() in rna_residues:
                        valid_rna_chains.append(chain.get_full_id()[2])
                        break   
        # valid_prot_chains = list(set(valid_prot_chains))
        # print(valid_prot_chains, valid_rna_chains)
        protein = proteins.ProteinInput.from_path(path, with_angles=False, return_dict=True, valid_chains=valid_prot_chains)
        rna = rnas.RNAInput.from_path(path, valid_rna_chains)
        complex_dict = complex_merge([protein[chain] for chain in valid_prot_chains], [rna[chain] for chain in valid_rna_chains])
        return self(**complex_dict)
    
    @property
    def length(self):
        return len(self.seq)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        texts = []
        texts += [f'seq: {self.seq}']
        texts += [f'length: {len(self.seq)}']
        texts += [f"mask: {''.join(self.mask.astype('int').astype('str'))}"]
        if self.chainid is not None:
            texts += [f"chainid: {''.join(self.chainid.astype('int').astype('str'))}"]
            texts += [f"identifier: {''.join(self.identifier.astype('int').astype('str'))}"]
        names = [
            'restype',
            'atom_mask',
            'atom_positions',
        ]
        for name in names:
            value = getattr(self, name)
            if value is None:
                text = f'{name}: None'
            else:
                text = f'{name}: {value.shape}'
            texts += [text]
        text = ', \n  '.join(texts)
        text = f'Protein-RNA Complex(\n  {text}\n)'
        return text
    
    
def complex_merge(protein, rna):
    assert len(protein) > 0
    assert len(rna) > 0

    p_lengths = [p.length for i, p in enumerate(protein)]
    r_lengths = [r.length for i, r in enumerate(rna)]
    lengths = p_lengths + r_lengths
    prot_list = []
    na_list = []
    seq = "".join([item.seq for item in protein + rna])        
    # identifier = np.array([0] * len(protein) + [1] * len(rna), dtype=np.int32)
    mask = np.concatenate([item.mask for item in protein + rna] )
    chain_arr = np.concatenate([[i] * p for i, p in enumerate(lengths)]).astype('int')
    res_nb = np.zeros([len(seq)])
    restype = np.zeros([len(seq)])
    atom_positions = np.zeros([len(seq), 37 + 27, 3])
    atom41_positions = np.zeros([len(seq), 14 + 27, 3])
    atom41_masks = np.zeros([len(seq), 14 + 27])
    atom_masks = np.zeros([len(seq), 37 + 27])
    identifier = np.zeros([len(seq)])
    curr_idx = 0
    for item in protein:
        prot_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.aatype
        identifier[curr_idx: curr_idx+item.length] = 0
        atom_positions[curr_idx: curr_idx+item.length, :37, :] = item.atom_positions
        atom14, mask_14, arrs = atom37_to_atom14(item.aatype, item.atom_positions, [item.atom_mask])
        mask_14 = arrs[0] * mask_14
        atom41_positions[curr_idx: curr_idx+item.length, :14, :] = atom14
        atom41_masks[curr_idx: curr_idx+item.length, :14] = mask_14
        atom_masks[curr_idx: curr_idx+item.length, :37] = item.atom_mask
        curr_idx += item.length
    for item in rna:
        na_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.basetype + 21
        identifier[curr_idx: curr_idx+item.length] = 1
        atom_positions[curr_idx: curr_idx+item.length, 37:, :] = item.atom_positions
        atom41_positions[curr_idx: curr_idx+item.length, 14:, :] = item.atom_positions
        atom41_masks[curr_idx: curr_idx+item.length, 14:] = item.atom_mask
        atom_masks[curr_idx: curr_idx+item.length, 37:] = item.atom_mask
        curr_idx += item.length

    complex_dict = {
        'seq': seq,
        'mask': mask,
        'restype': restype,
        'res_nb': res_nb,
        
        'prot_seqs': prot_list,
        'na_seqs': na_list,

        'atom_mask': atom_masks,
        'atom_positions': atom_positions,
        
        'atom41_mask': atom41_masks,
        'atom41_positions': atom41_positions,

        'identifier': identifier,
        'chainid': chain_arr
    }
    
    return complex_dict

if __name__ == '__main__':
    comp = ComplexInput.from_path('/home/HR/PIXberts/datasets/PNA350/PDBs_filtered/1RPU.pdb')
    print("Complex:", comp)
    print(comp.atom_positions[1])