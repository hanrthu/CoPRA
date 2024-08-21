
from ._base import register_transform
import torch
from data.rna import get_backbone_coords, RNA_ATOMS


@register_transform('select_atom')
class SelectAtom(object):

    def __init__(self, resolution):
        super().__init__()
        assert resolution in ('full', 'backbone', 'backbone+R1', 'C_Only')
        self.resolution = resolution

    def __call__(self, data):
        if self.resolution == 'full':
            data['pos_atoms'] = data['pos_heavyatom'][:, :]
            data['mask_atoms'] = data['mask_heavyatom'][:, :]
            
        elif self.resolution == 'backbone':
            pos_atoms = torch.zeros([data['pos_heavyatom'].shape[0]] + [4, 3], device=data['pos_heavyatom'].device)
            mask_atoms = torch.zeros([data['mask_heavyatom'].shape[0]] + [4], device=data['mask_heavyatom'].device).bool()
            # print(pos_atoms.shape, mask_atoms.shape)
            pos_atoms[data['identifier']==0] = data['pos_heavyatom'][data['identifier']==0, :4]
            mask_atoms[data['identifier']==0] = data['mask_heavyatom'][data['identifier']==0, :4]
            na_atoms = data['pos_heavyatom'][data['identifier']==1, 14:]
            na_seqs = data['seq']
            indices = torch.nonzero(data['identifier'])
            first_index = indices[0].item() if len(indices) > 0 else -1
            na_seqs = na_seqs[first_index:]
            fill_value = 1e-5
            pyrimidine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("C1'"), RNA_ATOMS.index("N1")]
            purine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("C1'"), RNA_ATOMS.index("N9")]
            backbone_coords = get_backbone_coords(na_atoms, na_seqs, pyrimidine_bb_indices, purine_bb_indices, fill_value)
            pos_atoms[data['identifier']==1, :4] = backbone_coords
            mask_atoms[data['identifier']==1, :4] = (backbone_coords != fill_value).sum(-1).bool()
            data['pos_atoms'] = pos_atoms
            data['mask_atoms'] = mask_atoms
            
        elif self.resolution == 'backbone+R1':
            # For Protein, it's N,CA,C,O,CB; For RNA, it's P, C4', N1/N9, C1'
            pos_atoms = torch.zeros([data['pos_heavyatom'].shape[0]] + [5, 3], device=data['pos_heavyatom'].device)
            mask_atoms = torch.zeros(data['mask_heavyatom'].shape[0] + [5], device=data['mask_heavyatom'].device).bool()
            pos_atoms[data['identifier']==0] = data['pos_heavyatom'][data['identifier']==0, :5]
            mask_atoms[data['identifier']==0] = data['mask_heavyatom'][data['identifier']==0, :5]
            
            na_atoms = data['pos_heavyatom'][data['identifier']==1, 14:]
            na_seqs = data['seq']
            indices = torch.nonzero(data['identifier'])
            first_index = indices[0].item() if len(indices) > 0 else -1
            na_seqs = na_seqs[first_index:]
            fill_value = 1e-5
            pyrimidine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("C1'"), RNA_ATOMS.index("C5'"), RNA_ATOMS.index("N1")]
            purine_bb_indices = [RNA_ATOMS.index("P"), RNA_ATOMS.index("C4'"), RNA_ATOMS.index("C1'"), RNA_ATOMS.index("C5'"), RNA_ATOMS.index("N9")]
            backbone_coords = get_backbone_coords(na_atoms, na_seqs, pyrimidine_bb_indices, purine_bb_indices, fill_value)
            pos_atoms[data['identifier']==1, :5] = backbone_coords
            mask_atoms[data['identifier']==1, :5] = backbone_coords != fill_value
            
            data['pos_atoms'] = data['pos_heavyatom'][:, :5]
            data['mask_atoms'] = data['mask_heavyatom'][:, :5]
            
        elif self.resolution == 'C_Only':
            data['pos_atoms'] = data['pos_heavyatom'][:, :1]
            data['mask_atoms'] = data['mask_heavyatom'][:, :1]
            
        return data
