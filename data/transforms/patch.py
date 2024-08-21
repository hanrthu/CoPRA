import random
import torch

from ._base import _index_select_data, register_transform, _get_CB_positions, _index_select_complex
import math

@register_transform('focused_random_patch')
class FocusedRandomPatch(object):

    def __init__(self, focus_attr, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.focus_attr = focus_attr
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        focus_flag = (data[self.focus_attr] > 0)  # (L, )
        if focus_flag.sum() == 0:
            # If there is no active residues, randomly pick one.
            focus_flag[random.randint(0, focus_flag.size(0) - 1)] = True
        seed_idx = torch.multinomial(focus_flag.float(), num_samples=1).item()

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        nbh_seed_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        core_idx = nbh_seed_idx[focus_flag[nbh_seed_idx]]  # (Ac, ), the core-set must be a subset of the focus-set
        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, ) # The distance to the itself is zero, thus the item must be chose.
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('random_patch')
class RandomPatch(object):

    def __init__(self, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        seed_idx = random.randint(0, data['aa'].size(0) - 1)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        core_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_with_padding_patch')
class SelectedRegionWithPaddingPatch(object):

    def __init__(self, select_attr, each_residue_nbh_size, patch_size_limit):
        super().__init__()
        self.select_attr = select_attr
        self.each_residue_nbh_size = each_residue_nbh_size
        self.patch_size_limit = patch_size_limit

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        pos_sel = pos_CB[select_flag]  # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel)  # (L, S)
        nbh_sel_idx = torch.argsort(dist_from_sel, dim=0)[:self.each_residue_nbh_size, :]  # (nbh, S)
        patch_idx = nbh_sel_idx.view(-1).unique()  # (patchsize,)

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_fixed_size_patch')
class SelectedRegionFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        pos_sel = pos_CB[select_flag]  # (S, 3)
        # print("Pos CB and sel:", pos_sel.shape, pos_CB.shape, select_flag.shape)
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]  # (L, )
        # print(self.patch_size)
        patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]

        data_patch = _index_select_data(data, patch_idx)
        return data_patch
    
@register_transform('selected_region_with_distmap')
class SelectedRegionWithDistmap(object):

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, data):
        atoms_dist_min = data['atom_min_dist']

        identifier = data['identifier']
        tmp = atoms_dist_min[identifier==0]
        interface_distance = tmp[:, identifier==1]
        prot_min_dist = interface_distance.min(dim=1)[0]
        rna_min_dist = interface_distance.transpose(0, 1).min(dim=1)[0]
        total_min = torch.cat([prot_min_dist, rna_min_dist], dim=0)
        patch_idx = torch.argsort(total_min)[:self.patch_size]
        patch_idx, _ = torch.sort(patch_idx)
        # print(self.patch_size)
        data_patch = _index_select_complex(data, patch_idx)
        data_patch['patch_idx'] = patch_idx
        return data_patch
    
@register_transform('interface_max_size_patch')
class InterfaceFixedMaxSizePatch(object):
    def __init__(self, max_size, interface_dist=7.5, kernel_size=100):
        super().__init__()
        self.max_size = max_size
        self.interface_dist = interface_dist
        self.kernel_size = kernel_size
    
    def segment_fill_zero(self, mask, chain_nb):
        unique_segs = torch.unique(chain_nb)
        result = mask.clone()
        can_fills = []
        for seg in unique_segs:
            seg_mask = chain_nb == seg
            seg_x = mask[seg_mask]
            
            ones_mask = seg_x == 1
            seg_result = ones_mask.clone()
            # cumsum = torch.cumsum(ones_mask, dim=0)
            indices = torch.nonzero(ones_mask).flatten()
            start = indices.min()
            end = indices.max() + 1
            seg_result[start: end] = True
            if len(seg_result) <= 30:
                seg_result[:] = True
            # seg_result = (cumsum > 0).bool()
            can_fill = (~seg_result).sum()
            can_fills.append(can_fill.item())
            result[seg_mask] = seg_result
        return result, can_fills
        
    def segment_fill_to_max(self, mask, chain_nb, length_to_fill, can_fill, prot_seqs, na_seqs):
        unique_segs = torch.unique(chain_nb)
        result = mask.clone()
        # assert sum(can_fill) >= length_to_fill
        remainders = length_to_fill
        prot_seqs_new = []
        na_seqs_new = []
        for i, seg in enumerate(unique_segs):
            to_fill = math.ceil(int((can_fill[i] / (sum(can_fill) + 1e-5)) * length_to_fill))
            if remainders < to_fill:
                to_fill = remainders
            remainders -= to_fill
            seg_mask = chain_nb == seg
            seg_x = mask[seg_mask]
            
            indices = torch.nonzero(seg_x).flatten()
            try:
                left_len = indices.min()
                right_len = len(seg_x) - indices.max() - 1
            except:
                print("???")
                
            if to_fill >= left_len + right_len:
                seg_x[:] = 1
                start = 0
                end = indices.max() + 1
            else: 
                left_fill = random.randint(max(to_fill-right_len, 0), min(left_len,to_fill))
                right_fill = to_fill - left_fill
                assert left_fill + right_fill == to_fill
                start = indices.min() - left_fill
                end = indices.max() + right_fill + 1
                seg_x[start: end] = 1
            
            result[seg_mask] = seg_x
            
            if i < len(prot_seqs):
                prot_seq_new = prot_seqs[i][start: end]
                prot_seqs_new.append(prot_seq_new)
            else:
                na_seq_new = na_seqs[i-len(prot_seqs)][start: end]
                na_seqs_new.append(na_seq_new)
        return result, prot_seqs_new, na_seqs_new
    
    def __call__(self, data):
        if len(data['pos_atoms']) <= self.max_size:
            # print("Short data, no need to process!")
            return data
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])
        identifier = data['identifier']
        chain_nb = data['chain_nb']
        dist_map = torch.cdist(pos_CB, pos_CB)
        dist_map[identifier[:, None] == identifier[None, :]] = 10000
        # print("Original data:", data['id'], len(''.join(data['prot_seqs'])), len(''.join(data['rna_seqs'])))
        contact_dis_min = torch.min(dist_map, dim=-1)[0]
        kernel_area = torch.zeros_like(contact_dis_min).bool()
        # kernel_area = contact_dis_min <= self.interface_dist
        contact_indices = torch.argsort(contact_dis_min)
        kernel_indices = contact_indices[: self.kernel_size]
        kernel_area.index_fill_(dim=0, index=kernel_indices, value=True)
        continuous_kernel_area, can_fill = self.segment_fill_zero(kernel_area, chain_nb)
        # if torch.sum(continuous_kernel_area)  > self.max_size:
        #     print("Max size:", torch.sum(continuous_kernel_area))
        to_fill = max((self.max_size - torch.sum(continuous_kernel_area)).item(), 0)
        continuous_kernel_area, prot_seq_new, na_seq_new = self.segment_fill_to_max(continuous_kernel_area, chain_nb, 
                                                                                    to_fill, can_fill,
                                                                                    data['prot_seqs'], data['rna_seqs'])
        select_idx = torch.nonzero(continuous_kernel_area).flatten()
        # print("New selected data:", len(select_idx))
        data = _index_select_complex(data, select_idx)
        data['prot_seqs'] = prot_seq_new
        data['rna_seqs'] = na_seq_new
        prot_lengths = [len(item) for item in prot_seq_new]
        na_lengths = [len(item) for item in na_seq_new]
        data['max_prot_length'] = max(prot_lengths)
        data['max_na_length'] = max(na_lengths)
        return data