from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import io
from typing import Dict, Optional, List
import numpy as np
import scipy
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from data.protein.residue_constants import restypes_with_x, restype_order_with_x, STANDARD_ATOM_MASK, restype_3to1, \
    restype_order, restype_num, atom_type_num, atom_types, atom_order, restype_lexico_order, residue_atoms
from data.protein.atom_convert import atom37_to_torsion7_np, atom37_to_atom14


@dataclass
class ProteinInput:
    seq: str  # L
    mask: np.ndarray  # (L, )
    aatype: np.ndarray  # (L, )
    aatype_lexico: np.ndarray # (L, )
    atom_mask: np.ndarray  # (L, 37)

    atom_positions: Optional[np.ndarray] = field(default=None)  # (L, 37, 3)
    residue_index: Optional[np.ndarray] = field(default=None)  # (L, )
    b_factors: Optional[np.ndarray] = field(default=None)  # (L, 37)

    torsion_angles: Optional[np.ndarray] = field(default=None) # (L, 7)
    alt_torsion_angles: Optional[np.ndarray] = field(default=None) # (L, 7)
    torsion_angles_sin_cos: Optional[np.ndarray] = field(default=None)  # (L, 7, 2)
    alt_torsion_angles_sin_cos: Optional[np.ndarray] = field(default=None)  # (L, 7, 2)
    torsion_angles_mask: Optional[np.ndarray] = field(default=None)  # (L, 7)
    res_nb: Optional[np.ndarray] = field(default=None) # (L, )
    resseq: Optional[np.ndarray] = field(default=None) # (L, )
    chain_nb: Optional[np.ndarray] = field(default=None) # (L, )
    chain_id: Optional[List] = field(default=None) # len(chain_id) = L
    icode: Optional[List] = field(default=None) # len(icode) = L

    chainid: Optional[np.ndarray] = field(default=None)  # (L)

    def __post_init__(self):
        assert not all([a is None for a in [self.seq, self.aatype]])
        if (self.seq is None) and (self.aatype is not None):
            self.seq = ''.join([restypes_with_x[a] for a in self.aatype.clip(0, 20)])
        if (self.seq is not None) and (self.aatype is None):
            self.aatype = np.array([restype_order_with_x.get(a, 21) for a in self.seq], dtype='int64')

        if self.atom_mask is None:
            self.atom_mask = STANDARD_ATOM_MASK[self.aatype].astype('bool')
        if self.mask is None:
            self.mask = self.atom_mask[:, 1].astype('bool')
        # print("AA type", self.aatype)
        # if not isinstance(self.aatype, np.ndarray):
        #     print("Exception！")
        #     # print(self)
        #     self.aatype = np.array(self.aatype)
        #     self.mask = np.array(self.mask)
        #     self.atom_mask = np.array(self.atom_mask)
        self.aatype = self.aatype.astype('int64')
        self.mask = self.mask.astype('bool')
        self.atom_mask = self.atom_mask.astype('bool')

        if self.b_factors is not None:
            if len(self.b_factors.shape) == 1:
                self.b_factors = self.b_factors[:, None] * self.atom_mask
            assert len(self.b_factors.shape) == 2

        assert len(self.mask.shape) == 1
        assert len(self.atom_mask.shape) == 2

    @property
    def length(self):
        return len(self.seq)

    @property
    def length_valid(self):
        return self.mask.sum().item()

    def has_structure(self):
        return (self.atom_positions is not None) and (self.mask.sum() != 0)

    def to_atom14(self):
        if self.atom_positions.shape[1] == 14:
            return self
        else:
            atom_positions, mask14, arrs = atom37_to_atom14(self.aatype, self.atom_positions,
                                                            [self.atom_mask, self.b_factors])
            atom_mask, b_factors = arrs
            atom_mask = atom_mask * mask14
            b_factors = b_factors * mask14
            return dataclasses.replace(self, atom_positions=atom_positions, atom_mask=atom_mask, b_factors=b_factors)

    def fillna(self, with_angles=True):
        length = len(self.seq)
        result = {}
        if self.atom_positions is None:
            result['atom_positions'] = np.zeros([length, 37, 3], dtype='float32')
        if self.residue_index is None:
            result['residue_index'] = np.arange(length).astype('int64')
        if self.b_factors is None:
            result['b_factors'] = np.zeros([length, 37], dtype='float32')
        if with_angles:
            if self.atom_positions is None:
                result_t = dict(
                    torsion_angles_sin_cos=np.zeros([length, 7, 2], dtype='float32'),
                    alt_torsion_angles_sin_cos=np.zeros([length, 7, 2], dtype='float32'),
                    torsion_angles_mask=np.zeros([length, 7], dtype='bool'),
                )
            elif self.torsion_angles_sin_cos is None:
                angles, alt_angles, sin_cos, sin_cos_mask, alt_sin_cos = atom37_to_torsion7_np(self.aatype, self.atom_positions,
                                                                           self.atom_mask)
                result_t = {
                    'torsion_angles': angles,
                    'alt_torsion_angles': alt_angles,
                    'torsion_angles_sin_cos': sin_cos,
                    'alt_torsion_angles_sin_cos': alt_sin_cos,
                    'torsion_angles_mask': sin_cos_mask,
                }
            else:
                result_t = {}
            result.update(result_t)
        return dataclasses.replace(self, **result)

    def padding(self, pad_width):
        if pad_width > 0:
            values_base = dict(
                seq=self.seq + 'X' * pad_width,
                mask=np.pad(self.mask, ((0, pad_width)), mode='constant', constant_values=False),
                aatype=np.pad(self.aatype, ((0, pad_width)), mode='constant', constant_values=21),
                atom_mask=np.pad(self.atom_mask, ((0, pad_width), (0, 0)), mode='constant', constant_values=False),
            )
            array_padding_settings = {
                'atom_positions': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'residue_index': dict(pad_width=((0, pad_width)), mode='edge'),
                'b_factors': dict(pad_width=((0, pad_width), (0, 0)), mode='edge'),
                'torsion_angles_sin_cos': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'alt_torsion_angles_sin_cos': dict(pad_width=((0, pad_width), (0, 0), (0, 0)), mode='edge'),
                'torsion_angles_mask': dict(pad_width=((0, pad_width), (0, 0)), mode='edge'),
                'chainid': dict(pad_width=((0, pad_width)), mode='constant', constant_values=20),
            }
            values = {}
            for k in array_padding_settings.keys():
                value = getattr(self, k)
                if value is not None:
                    value = np.pad(value, **array_padding_settings[k])
                    values[k] = value
            values_all = values_base
            values_all.update(values)
            return dataclasses.replace(self, **values_all)
        else:
            return self

    def slice(self, begin, end):
        if (begin < 0):
            raise Exception(f'error begin: {begin}')
        if end > self.length:
            pad_width = end - self.length
            self = self.padding(pad_width)
        names = set(self.__dataclass_fields__.keys())
        result = {}
        for name in names:
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = value[begin:end]
        return dataclasses.replace(self, **result)

    def mask_select(self, mask):
        mask = mask.astype('bool')
        # print(len(mask), self.length)
        assert len(mask) == self.length

        names = list(self.__dataclass_fields__.keys())
        result = {}
        for name in names:
            value = getattr(self, name)
            if name == 'seq':
                result[name] = ''.join([r for r, m in zip(value, mask) if m == 1])
            elif value is None:
                result[name] = value
            elif name == 'chain_id' or name == 'icode':
                result[name] = [r for r,m in zip(value, mask) if m == 1]
            else:
                # print("Else:", mask.shape, value, name)
                result[name] = value[mask]
        return dataclasses.replace(self, **result)

    def __getitem__(self, key):
        if isinstance(key, slice):
            begin = key.start if key.start else 0
            end = key.stop if key.stop else self.length
            if (begin < 0) or (end > self.length):
                raise Exception(f'error span: {key}')
            return self.slice(begin, end)
        else:
            raise TypeError('Index must be slice, not {}'.format(type(key).__name__))

    def append(self, other):
        # Append a new ProteinInput
        result = dict(
            seq=self.seq + other.seq,
            mask=np.concatenate([self.mask, other.mask]),
            aatype=np.concatenate([self.aatype, other.aatype]),
            atom_mask=np.concatenate([self.atom_mask, other.atom_mask]),
        )
        names = set(self.__dataclass_fields__.keys())
        for name in names:
            if name in result:
                continue
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = np.concatenate([value, getattr(other, name)])
        return dataclasses.replace(self, **result)

    def append_list(self, others):
        # Append ProteinInput lists
        if len(others) == 0:
            return self
        result = dict(
            seq=''.join([self.seq] + [other.seq for other in others]),
            mask=np.concatenate([self.mask] + [other.mask for other in others]),
            aatype=np.concatenate([self.aatype] + [other.aatype for other in others]),
            atom_mask=np.concatenate([self.atom_mask] + [other.atom_mask for other in others]),
        )
        names = set(self.__dataclass_fields__.keys())
        for name in names:
            if name in result:
                continue
            value = getattr(self, name)
            if value is None:
                result[name] = value
            else:
                result[name] = np.concatenate([value] + [getattr(other, name) for other in others])
        return dataclasses.replace(self, **result)

    @classmethod
    def from_pdbchain(cls, chain, idx):
        result = chain2arrays(chain, idx)
        return cls(**result)

    @classmethod
    def from_path(cls, path, with_angles=True, return_dict=False, verbose=False, valid_chains=None):
        # print(path)
        if isinstance(path, io.IOBase):
            file_string = path.read()
        else:
            # print(path)
            path = Path(path)
            file_string = path.read_text()

        proteins = {}
        if '.pdb' in str(path):
            chains = chains_from_pdb_string(file_string)
        elif '.cif' in str(path):
            chains = chains_from_cif_string(file_string)
        # print(path)
        chains.sort(key=lambda c: c.get_id())
        # print(chains)
        for idx, chain in enumerate(chains):
            # try:
            chainid = chain.get_full_id()[2]
            if valid_chains is not None and str(chainid) not in valid_chains:
                # print("!!!")
                # print("Removing chains:", chain, valid_chains)
                continue 
            # check the chains to avoid it only containing HOH
            accept = 0
            for residue in chain:
                if residue.get_resname() in residue_atoms.keys():
                    accept = 1
                    break
            if accept == 0:
                # print("Reject:", chain)
                continue
            protein = cls.from_pdbchain(chain, idx)
            if with_angles:
                protein = protein.fillna(with_angles=True)
            # print("Protein:", protein)
            proteins[chainid] = protein
            # except Exception as e:
            #     if verbose:
            #         print(path)
            #         print(chain)
            #         print(e)

        if not return_dict:
            protein = next(iter(proteins.values()))
            return protein
        else:
            return proteins

    def to_dict(self):
        return dataclasses.asdict(self)

    def get_center(self):
        atom_positions = self.atom_positions
        atom_mask = self.atom_mask
        if atom_mask is None:
            center = np.zeros(3)
        elif atom_mask.sum() == 0:
            center = np.zeros(3)
        else:
            center = atom_positions[atom_mask].mean(axis=0)
        return center

    def translation(self, translation):
        atom_positions = self.atom_positions
        atom_positions = atom_positions + translation[None, None, :]
        return dataclasses.replace(self, atom_positions=atom_positions)

    def rotation(self, rotation, around_point=None):
        atom_positions = self.atom_positions
        r = scipy.spatial.transform.Rotation.from_matrix(rotation)
        if around_point is not None:
            atom_positions_0 = atom_positions - around_point[None, None, :]
            atom_positions_new = r.apply(atom_positions_0.reshape(-1, 3)).reshape(-1, 37, 3)
            atom_positions_new = atom_positions_new + around_point[None, None, :]
        else:
            atom_positions_new = r.apply(atom_positions.reshape(-1, 3)).reshape(-1, 37, 3)

        return dataclasses.replace(self, atom_positions=atom_positions_new)

    def get_i_from_residue_index(self, index, around=False):
        i = np.searchsorted(self.residue_index, index)
        if i >= self.length:
            return -1
        elif index != self.residue_index[i] and (not around):
            return -1
        else:
            return i

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
            'aatype',
            'atom_mask',
            'atom_positions',
            'residue_index',
            'b_factors',
            'torsion_angles_sin_cos',
            'alt_torsion_angles_sin_cos',
            'torsion_angles_mask',
        ]
        for name in names:
            value = getattr(self, name)
            if value is None:
                text = f'{name}: None'
            else:
                text = f'{name}: {value.shape}'
            texts += [text]
        text = ', \n  '.join(texts)
        text = f'Protein(\n  {text}\n)'
        return text


def proteins_merge(proteins, chainids=None):
    assert len(proteins) > 0
    p = proteins[0]
    p = p.append_list(proteins[1:])
    if chainids is None:
        chain_arr = np.concatenate([[i] * p.length for i, p in enumerate(proteins)]).astype('int')
    else:
        chain_arr = np.concatenate([[i] * p.length for i, p in zip(chainids, proteins)]).astype('int')
    p = dataclasses.replace(p, chainid=chain_arr)
    return p


def seq2aatype(seq):
    aatype = np.array([restype_order_with_x.get(a, 21) for a in seq], dtype='int64')
    return aatype


def chains_from_pdb_string(pdb_str: str):
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)


    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    # if len(models) != 1:
    #     # raise ValueError(
    #     #     f"Only single model PDBs are supported. Found {len(models)} models."
    #     # )
    #     print(f"Warning: Only single model PDBs are recommended. Found {len(models)} models.")
    model = models[0]
    chains = list(model.get_chains())
    return chains

def chains_from_cif_string(cif_str: str):
    cif_fh = io.StringIO(cif_str)
    parser = MMCIFParser(QUIET=True)
    
    structure = parser.get_structure(None, cif_fh)
    models = list(structure.get_models())
    # if len(models) != 1:
    #     raise ValueError(
    #         f"Only single model PDBs are supported. Found {len(models)} models."
    #     )
    model = models[0]
    chains = list(model.get_chains())
    return chains

def chain2arrays(chain, idx):
    seq = []
    atom_positions = []
    aatype = []
    aatype_lexico = []
    atom_mask = []
    residue_index = []
    b_factors = []
    icodes = [] # residue icodes
    chain_id = [] # chain id in str
    chain_nb = []
    resseq = [] # resseq_idx
    res_nb = [] # resnumber_idx in the file
    seq_this = 0
    for res in chain:
        if res.get_resname() == 'HOH':
            continue
        if res.id[0] != ' ' or res.id[2] != ' ': # ignore abnormal hetfield and icode
            continue
        icode = res.get_id()[2]
        resseq_this = int(res.get_id()[1])
        res_shortname = restype_3to1.get(res.get_resname(), 'X')
        restype_idx = restype_order.get(
            res_shortname, restype_num)
        restype_idx_lexico = restype_lexico_order(restype_idx)
        pos = np.zeros((atom_type_num, 3))
        mask = np.zeros((atom_type_num,))
        res_b_factors = np.zeros((atom_type_num,))
        
        for atom in res:
            if atom.name not in atom_types:
                continue
            pos[atom_order[atom.name]] = atom.coord
            mask[atom_order[atom.name]] = 1.
            res_b_factors[atom_order[atom.name]] = atom.bfactor
            
        seq.append(res_shortname)
        aatype.append(restype_idx)
        aatype_lexico.append(restype_idx_lexico)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        chain_id.append(chain.get_full_id()[2])
        chain_nb.append(idx)
        b_factors.append(res_b_factors)

        if np.sum(mask) < 0.5:
            continue
        
        if seq_this == 0:
            seq_this = 1
        else:
            d_CA_CA = np.linalg.norm(atom_positions[-2][1] - atom_positions[-1][1], ord=2)
            if d_CA_CA <= 4.0:
                seq_this += 1
            else:
                d_resseq = resseq_this - resseq[-1]
                seq_this += max(2, d_resseq)
        resseq.append(resseq_this)
        res_nb.append(seq_this)
        icodes.append(icode)

    seq = ''.join(seq)
    atom_positions = np.array(atom_positions)
    atom_mask = np.array(atom_mask)
    aatype = np.array(aatype)
    aatype_lexico = np.array(aatype_lexico)
    residue_index = np.array(residue_index)
    b_factors = np.array(b_factors)
    chain_nb = np.array(chain_nb)
    # print("Atom Mask:", atom_mask.shape, chain_id)
    mask = atom_mask[:, 1] * np.array([r != 'X' for r in seq])
    resseq = np.array(resseq)
    res_nb = np.array(res_nb)
    result = {
        'seq': seq,
        'mask': mask,
        'aatype': aatype,
        'aatype_lexico': aatype_lexico,
        'atom_mask': atom_mask,
        'atom_positions': atom_positions,
        'residue_index': residue_index,
        'b_factors': b_factors,
        'res_nb': res_nb,
        'resseq': resseq,
        'chain_nb': chain_nb,
        'chain_id': chain_id,
        'icode': icodes
    }
    # print(result)
    return result

if __name__ == '__main__':
    import pickle
    pdbs = ProteinInput.from_pdb(io.StringIO('/home/rhan21/Research/DrugDD/UniPPI/datasets/SKEMPIv2/PDBs/1EFN.pdb'),
                                          with_angles=True, return_dict=True)
    chain_a = pdbs['A'].to_atom14()
    with open('protein_input.pkl', 'wb') as f:
        pickle.dump(pdbs, f)
    f.close()
    # print(np.where(chain_a.torsion_angles[0:30, 2] > 0, chain_a.torsion_angles[0:30, 2] - np.pi, chain_a.torsion_angles[0:30, 2] + np.pi)) # w, phi, psi, chi4



    # with open('../../protein_input.pkl', 'rb') as f:
    #     protein_input = pickle.load(f)
    # with open('../../parser.pkl', 'rb') as f:
    #     parser_input = pickle.load(f)
    # chain_a = protein_input['A'].to_atom14()
    # omega_pro = chain_a.torsion_angles[:, 0]
    # phi_pro = chain_a.torsion_angles[:, 1]
    # psi_pro = chain_a.torsion_angles[:, 2]
    # chis = chain_a.torsion_angles[:, 3:]
    #
    # chain_info = "".join(parser_input.chain_id)
    # index = chain_info.rfind('A')
    # phi_par = parser_input.phi[0: index+1]
    # psi_par = parser_input.psi[0: index+1]
    # chis_par = parser_input.chi[0: index+1]
    #
    # difference_phi = (phi_pro - phi_par.numpy()).sum()
    # difference_chi = (chis - chis_par.numpy()).sum()
    # difference_psi = (psi_pro - psi_par.numpy()).sum()
    # pos_pro = chain_a.atom_positions * chain_a.atom_mask[..., None]
    # pos_par = parser_input.pos_heavyatom[0: index + 1, :-1, :].numpy()
    # difference_pos = (pos_pro - pos_par).sum() / len(pos_pro)
    # difference_pos_mask = (chain_a.atom_mask != parser_input.mask_heavyatom[0: index+1, :-1].numpy()).sum()
    # import torch
    #
    # torsion_par = torch.cat(
    #     [parser_input.phi_mask.unsqueeze(-1), parser_input.psi_mask.unsqueeze(-1), parser_input.chi_mask],
    #     dim=-1).numpy()[0: index + 1]
    #
    # difference_torsion_mask = (chain_a.torsion_angles_mask[:, 1:] != torsion_par).sum()
    # aa_pro = chain_a.aatype_lexico
    # aa_par = parser_input.aa.numpy()[0:index+1]
    # diff_aa = (aa_pro != aa_par).sum()
    # bf_pro = chain_a.b_factors
    # bf_par = parser_input.bfactor_heavyatom.numpy()[0:index+1,:-1]
    # diff_bf = (bf_pro != bf_par).sum()
    # alt_par = parser_input.chi_alt.numpy()[0:index + 1]
    # alt_pro = chain_a.alt_torsion_angles[..., 3:]
    # diff_alt = (alt_pro - alt_par).sum()
    # print("Hello")