from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.append('/home/HR/PIXberts/')
import io
from typing import Dict, Optional, List
import numpy as np
import scipy
from data.rna.data_utils import pdb_to_array, get_backbone_coords, cif_to_array


@dataclass
class RNAInput:
    seq: str  # L
    mask: np.ndarray  # (L, )
    basetype: np.ndarray  # (L, )
    atom_mask: np.ndarray  # (L, 27)

    atom_positions: Optional[np.ndarray] = field(default=None)  # (L, 27, 3)
    residue_index: Optional[np.ndarray] = field(default=None)  # (L, )

    res_nb: Optional[np.ndarray] = field(default=None) # (L, )
    resseq: Optional[np.ndarray] = field(default=None) # (L, )
    chain_nb: Optional[np.ndarray] = field(default=None) # (L, )
    chain_id: Optional[List] = field(default=None) # len(chain_id) = L

    chainid: Optional[np.ndarray] = field(default=None)  # (L)

   
    @classmethod
    def from_path(self, pdb_filepath, valid_chains):
        """
        Load RNA backbone from PDB file.

        Args:
            pdb_filepath (str): Path to PDB file.
        """
        pdb_filepath = str(pdb_filepath)
        if '.pdb' in pdb_filepath:
            rnas = pdb_to_array(
                pdb_filepath, valid_chains, return_sec_struct=True, return_sasa=False)
        else:
            rnas = cif_to_array(pdb_filepath, valid_chains)
        # print(rnas)
        rna_dict = {}
        for key in rnas:
            rna = self(**rnas[key])
            rna_dict[key] = rna
        return rna_dict
        # coords = get_backbone_coords(coords, sequence)
        # rna = {
        #     'sequence': sequence,
        #     'coords_list': coords,
        # }
        # return rna
    
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

        names = [
            'basetype',
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
        text = f'RNA(\n  {text}\n)'
        return text
    
    
if __name__ == '__main__':
    rna = RNAInput.from_path('/home/HR/PIXberts/datasets/BioLiP2/PDBs/1A1T_A_B.cif', valid_chains=['B'])
    # print(rna["B"])