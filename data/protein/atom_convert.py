import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.protein import residue_constants
from utils.geometry import apply_rigid, multi_rigid, get_dih

chi_atom_indices_ref = [
    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 11],
     [3, 5, 11, 23],
     [5, 11, 23, 32]],

    [[0, 1, 3, 5],
     [1, 3, 5, 16],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 16],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 10],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 11],
     [3, 5, 11, 26],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 11],
     [3, 5, 11, 26],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 14],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 6],
     [1, 3, 6, 12],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 12],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 11],
     [3, 5, 11, 19],
     [5, 11, 19, 35]],

    [[0, 1, 3, 5],
     [1, 3, 5, 18],
     [3, 5, 18, 19],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 12],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 11],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 8],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 9],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 12],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 5],
     [1, 3, 5, 12],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 1, 3, 6],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],

    [[0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
]
chi_atom_indices_ref = np.array(chi_atom_indices_ref)
chi_angles_mask_ref = np.array(residue_constants.chi_angles_mask)
chi_pi_periodic_ref = np.array(residue_constants.chi_pi_periodic)
restype_rigid_group_default_frame = residue_constants.restype_rigid_group_default_frame
restype_atom37_to_rigid_group = residue_constants.restype_atom37_to_rigid_group
restype_atom37_rigid_group_positions = residue_constants.restype_atom37_rigid_group_positions

restype_name_to_atom14_names = residue_constants.restype_name_to_atom14_names
atom_order = residue_constants.atom_order
restype_order_with_x = residue_constants.restype_order_with_x
restype_3to1 = residue_constants.restype_3to1


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def _make_atom37_to_atom14():
    restype_atom37_to_atom14_index = np.zeros([21, 14], dtype='long')
    restype_atom37_to_atom14_mask = np.zeros([21, 14], dtype='bool')
    for res3, atoms in restype_name_to_atom14_names.items():
        res = restype_3to1.get(res3, 'X')
        i = restype_order_with_x.get(res, 20)
        for j, atom in enumerate(atoms):
            if atom != '':
                j_atom37 = atom_order[atom]
                restype_atom37_to_atom14_index[i, j] = j_atom37
                restype_atom37_to_atom14_mask[i, j] = True
    return restype_atom37_to_atom14_index, restype_atom37_to_atom14_mask


restype_atom37_to_atom14_index, restype_atom37_to_atom14_mask = _make_atom37_to_atom14()


def atom37_to_atom14(aatype, all_atom_pos, arrs=[]):
    atom_indices = restype_atom37_to_atom14_index[aatype]
    mask14 = restype_atom37_to_atom14_mask[aatype]
    pos14 = np.take_along_axis(all_atom_pos, atom_indices[..., None], axis=1)
    arrs = [np.take_along_axis(arr, atom_indices, axis=1) for arr in arrs]
    return pos14, mask14, arrs

def alt(angles):
    return np.where(angles > 0, angles - np.pi, (angles + np.pi) % np.pi)

def atom37_to_torsion7_np(aatype, all_atom_pos, all_atom_mask):
    '''
    aatype: np.array[L,]
    all_atom_pos: np.array[L,37,3]
    all_atom_mask: np.array[L,37]
    '''
    all_atom_pos_prev = np.roll(all_atom_pos, shift=1, axis=0)
    all_atom_pos_prev[0] = 0

    all_atom_mask_prev = np.roll(all_atom_mask, shift=1, axis=0)
    all_atom_mask_prev[0] = 0

    all_atom_pos_post = np.roll(all_atom_pos, shift=-1, axis=0)
    all_atom_pos_post[-1] = 0

    all_atom_mask_post = np.roll(all_atom_mask, shift=-1, axis=0)
    all_atom_mask_post[-1] = 0

    atom_indices = chi_atom_indices_ref[aatype]

    # Torsion
    pre_omega_atom_pos = np.concatenate([all_atom_pos_prev[..., [1, 2], :], all_atom_pos[..., [0, 1], :]],
                                        axis=-2)  # pre omega
    phi_atom_pos = np.concatenate([all_atom_pos_prev[..., [2], :], all_atom_pos[..., [0, 1, 2], :]], axis=-2)  # phi
    psi_atom_pos = np.concatenate([all_atom_pos[..., [0, 1, 2], :], all_atom_pos_post[..., [0], :]], axis=-2)  #
    chis_atom_pos = np.take_along_axis(all_atom_pos, atom_indices.reshape(-1, 16)[..., None], axis=1).reshape(-1, 4, 4,
                                                                                                              3)
    pos = np.concatenate([
        pre_omega_atom_pos[:, None],
        phi_atom_pos[:, None],
        psi_atom_pos[:, None],
        chis_atom_pos,
    ], axis=1)
    pos = torch.tensor(pos)
    angles = get_dih(pos[:, :, 0], pos[:, :, 1], pos[:, :, 2], pos[:, :, 3])  # torch.tensor
    angles = angles.numpy()
    # Alt torsion
    chi_is_ambiguous = chi_pi_periodic_ref[aatype]
    mirror_torsion_angles = np.ones_like(angles)
    mirror_torsion_angles[..., 3:] = 1.0 - 2.0 * chi_is_ambiguous
    alt_chis = np.where(chi_is_ambiguous, alt(angles[..., 3:]), angles[..., 3:])
    alt_angles = np.copy(angles)
    alt_angles[..., 3:] = alt_chis
    alt_angles[np.isnan(alt_angles)] = 0
    # mask
    pre_omega_atom_mask = np.concatenate([all_atom_mask_prev[..., [1, 2]], all_atom_mask[..., [0, 1]]],
                                         axis=-1)  # pre omega
    phi_atom_mask = np.concatenate([all_atom_mask_prev[..., [2]], all_atom_mask[..., [0, 1, 2]]], axis=-1)  # phi
    psi_atom_mask = np.concatenate([all_atom_mask[..., [0, 1, 2]], all_atom_mask_post[..., [0]]], axis=-1)
    chis_atom_mask = np.take_along_axis(all_atom_mask, atom_indices.reshape(-1, 16), axis=1).reshape(-1, 4, 4)
    chi_angles_mask = chi_angles_mask_ref[aatype]
    chis_atom_mask = chis_atom_mask.prod(axis=-1) * chi_angles_mask
    mask = np.concatenate([
        pre_omega_atom_mask.prod(axis=-1)[:, None],
        phi_atom_mask.prod(axis=-1)[:, None],
        psi_atom_mask.prod(axis=-1)[:, None],
        chis_atom_mask,
    ], axis=1)
    mask = mask.astype('bool')  # np.array

    angles[~mask] = 0
    angles[np.isnan(angles)] = 0
    sin = np.sin(angles)
    cos = np.cos(angles)
    sin_cos = np.stack([sin, cos], axis=-1)
    # sin_cos[..., 2, :] = sin_cos[..., 2, :] * (-1) # This is a way to approximate Psi, originally used N, Ca, C, O, now already calibrated.
    alt_sin_cos = sin_cos * mirror_torsion_angles[..., None]
    return angles, alt_angles, sin_cos, mask, alt_sin_cos


def torsion7_to_frame8_np(aatype, sin_cos):
    '''
    aatype: np.array[L]
    sin_cos: np.array[L,7,2]
    '''
    default_frames = restype_rigid_group_default_frame[aatype]
    default_rot = default_frames[..., :3, :3]
    default_trans = default_frames[..., :3, 3]

    sin_angles = np.concatenate([np.zeros_like(aatype)[..., None], sin_cos[..., 0]], axis=-1)
    cos_angles = np.concatenate([np.ones_like(aatype)[..., None], sin_cos[..., 1]], axis=-1)
    zeros = np.zeros_like(sin_angles)
    ones = np.ones_like(sin_angles)
    all_rots = np.stack([ones, zeros, zeros,
                         zeros, cos_angles, -sin_angles,
                         zeros, sin_angles, cos_angles], axis=-1)
    all_rots = all_rots.reshape(-1, 8, 3, 3)  # Rotation based on anles

    r, t = multi_rigid(default_rot, default_trans, all_rots, np.zeros_like(default_trans))

    r_chi1, t_chi1 = r[..., 4, :, :], t[..., 4, :]
    r_chi2, t_chi2 = multi_rigid(r_chi1, t_chi1, r[..., 5, :, :], t[..., 5, :])
    r_chi3, t_chi3 = multi_rigid(r_chi2, t_chi2, r[..., 6, :, :], t[..., 6, :])
    r_chi4, t_chi4 = multi_rigid(r_chi3, t_chi3, r[..., 7, :, :], t[..., 7, :])

    r = np.concatenate([
        r[..., :5, :, :], r_chi2[..., None, :, :], r_chi3[..., None, :, :], r_chi4[..., None, :, :]
    ], axis=-3)
    t = np.concatenate([
        t[..., :5, :], t_chi2[..., None, :], t_chi3[..., None, :], t_chi4[..., None, :]
    ], axis=-2)
    return r, t


def local_frame8_to_local_atom37_np(aatype, r, t):
    '''
    aatype: np.array[L]
    r: np.array[L,3,3]
    t: np.array[L,3]
    '''
    residx_to_group_idx = restype_atom37_to_rigid_group[aatype]
    lit_positions = restype_atom37_rigid_group_positions[aatype]
    r2 = np.take_along_axis(r, residx_to_group_idx[..., None, None], axis=-3)
    t2 = np.take_along_axis(t, residx_to_group_idx[..., None], axis=-2)
    # out = apply_rigid(lit_positions, r2, t2)
    dims = list(range(len(r2.shape)))
    dims[-1], dims[-2] = dims[-2], dims[-1]
    r2 = r2.transpose(*dims)  # Need to transpose
    out = apply_rigid(lit_positions, r2, t2)
    return out


def torsion7_to_local_atom37_np(aatype, sin_cos):
    r, t = torsion7_to_frame8_np(aatype, sin_cos)
    atom_positions = local_frame8_to_local_atom37_np(aatype, r, t)
    return atom_positions, t


class TorsionPositionTransformer(nn.Module):
    def __init__(self):
        super(TorsionPositionTransformer, self).__init__()
        self.chi_atom_indices_ref_pt = nn.Parameter(torch.tensor(chi_atom_indices_ref), requires_grad=False)
        self.chi_angles_mask_pt = nn.Parameter(torch.tensor(chi_angles_mask_ref), requires_grad=False)
        self.chi_pi_periodic_pt = nn.Parameter(torch.tensor(chi_pi_periodic_ref).float(), requires_grad=False)
        self.restype_rigid_group_default_frame_torch = nn.Parameter(torch.tensor(restype_rigid_group_default_frame),
                                                                    requires_grad=False)
        self.restype_atom37_to_rigid_group_torch = nn.Parameter(torch.tensor(restype_atom37_to_rigid_group),
                                                                requires_grad=False)
        self.restype_atom37_rigid_group_positions_torch = nn.Parameter(
            torch.tensor(restype_atom37_rigid_group_positions).float(), requires_grad=False)

        self.restype_atom37_to_atom14_index_torch = nn.Parameter(torch.tensor(restype_atom37_to_atom14_index).long(),
                                                                 requires_grad=False)
        self.restype_atom37_to_atom14_mask_torch = nn.Parameter(torch.tensor(restype_atom37_to_atom14_mask).bool(),
                                                                requires_grad=False)

    def atom37_to_torsion7_torch(self, aatype, all_atom_pos, all_atom_mask):
        '''
        aatype: torch.tensor[B,L,]
        all_atom_pos: torch.tensor[B,L,37,3]
        all_atom_mask: torch.tensor[B,L,37]
        '''
        all_atom_pos_prev = torch.roll(all_atom_pos, shifts=1, dims=1)
        all_atom_pos_prev[:, 0] = 0

        all_atom_mask_prev = torch.roll(all_atom_mask, shifts=1, dims=1)
        all_atom_mask_prev[:, 0] = 0
        atom_indices = self.chi_atom_indices_pt[aatype]  # [B,L,4,4]

        # Torsion
        pre_omega_atom_pos = torch.cat([all_atom_pos_prev[..., [1, 2], :], all_atom_pos[..., [0, 1], :]],
                                       dim=-2)  # pre omega # prev CA, C; this N, CA
        phi_atom_pos = torch.cat([all_atom_pos_prev[..., [2], :], all_atom_pos[..., [0, 1, 2], :]],
                                 dim=-2)  # phi # prev C; this N, CA, C
        psi_atom_pos = torch.cat([all_atom_pos[..., [0, 1, 2, 4], :]], dim=-2)  # psi is minus # this N, CA, C, O
        indices_temp = atom_indices.flatten(-2, -1)[..., None].expand(
            [-1] * (len(aatype.shape) + 1) + [3])
        chis_atom_pos = torch.gather(all_atom_pos, index=indices_temp, dim=-2).unflatten(dim=-2, sizes=(4, 4))
        pos = torch.cat([
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ], dim=-3)
        angles = get_dih(pos[..., 0, :], pos[..., 1, :], pos[..., 2, :], pos[..., 3, :])  # torch.tensor

        # Alt torsion
        chi_is_ambiguous = self.chi_pi_periodic_pt[aatype]
        mirror_torsion_angles = torch.ones_like(angles)
        mirror_torsion_angles[..., 3:] = 1.0 - 2.0 * chi_is_ambiguous

        # mask
        pre_omega_atom_mask = torch.cat([all_atom_mask_prev[..., [1, 2]], all_atom_mask[..., [0, 1]]],
                                        dim=-1)  # pre omega
        phi_atom_mask = torch.cat([all_atom_mask_prev[..., [2]], all_atom_mask[..., [0, 1, 2]]], dim=-1)  # phi
        psi_atom_mask = torch.cat([all_atom_mask[..., [0, 1, 2, 4]]], dim=-1)  # psi
        chis_atom_mask = torch.gather(all_atom_mask, index=atom_indices.flatten(-2, -1), dim=-1).unflatten(dim=-1,
                                                                                                           sizes=(4, 4))
        chi_angles_mask = self.chi_angles_mask_pt[aatype]
        chis_atom_mask = chis_atom_mask.prod(dim=-1) * chi_angles_mask
        mask = torch.cat([
            pre_omega_atom_mask.prod(dim=-1)[..., None],
            phi_atom_mask.prod(dim=-1)[..., None],
            psi_atom_mask.prod(dim=-1)[..., None],
            chis_atom_mask,
        ], dim=-1)
        mask = mask.bool()  # np.array

        angles[~mask] = 0
        angles[torch.isnan(angles)] = 0
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        sin_cos = torch.stack([sin, cos], dim=-1)
        sin_cos[..., 2, :] = sin_cos[..., 2, :] * (-1)
        alt_sin_cos = sin_cos * mirror_torsion_angles[..., None]
        return sin_cos, mask, alt_sin_cos

    def torsion7_to_frame8_torch(self, aatype: torch.Tensor, sin_cos: torch.Tensor):
        '''
        aatype: torch.tensor[B,L]
        sin_cos: torch.tensor[B,L,7,2]
        '''
        default_frames = self.restype_rigid_group_default_frame_torch[aatype]  # [B, L, 8, 4, 4]
        default_rot = default_frames[..., :3, :3]
        default_trans = default_frames[..., :3, 3]

        sin_angles = torch.cat([torch.zeros_like(aatype)[..., None], sin_cos[..., 0]], dim=-1)
        cos_angles = torch.cat([torch.ones_like(aatype)[..., None], sin_cos[..., 1]], dim=-1)
        zeros = torch.zeros_like(sin_angles)
        ones = torch.ones_like(sin_angles)
        all_rots = torch.stack([ones, zeros, zeros,
                                zeros, cos_angles, -sin_angles,
                                zeros, sin_angles, cos_angles], dim=-1)
        all_rots = all_rots.unflatten(-1, sizes=(3, 3))

        r, t = multi_rigid(default_rot, default_trans, all_rots, torch.zeros_like(default_trans))

        r_chi1, t_chi1 = r[..., 4, :, :], t[..., 4, :]
        r_chi2, t_chi2 = multi_rigid(r_chi1, t_chi1, r[..., 5, :, :], t[..., 5, :])
        r_chi3, t_chi3 = multi_rigid(r_chi2, t_chi2, r[..., 6, :, :], t[..., 6, :])
        r_chi4, t_chi4 = multi_rigid(r_chi3, t_chi3, r[..., 7, :, :], t[..., 7, :])

        r = torch.cat([
            r[..., :5, :, :], r_chi2[..., None, :, :], r_chi3[..., None, :, :], r_chi4[..., None, :, :]
        ], dim=-3)
        t = torch.cat([
            t[..., :5, :], t_chi2[..., None, :], t_chi3[..., None, :], t_chi4[..., None, :]
        ], dim=-2)
        return r, t

    def ortho6ds_to_frame8_torch(self, aatype: torch.Tensor, ortho6ds: torch.Tensor):
        '''
        aatype: torch.tensor[B,L]
        sin_cos: torch.tensor[B,L,7,6]
        '''
        device = aatype.device
        default_frames = self.restype_rigid_group_default_frame_torch[aatype]  # [B, L, 8, 4, 4]
        default_rot = default_frames[..., :3, :3]
        default_trans = default_frames[..., :3, 3]

        ortho6ds_id = torch.tensor([1., 0., 0., 0., 1., 0.], device=device)
        ortho6ds_id = ortho6ds_id[None, None, None, :].expand(list(ortho6ds.shape[:-2]) + [1, 6])
        ortho6ds = torch.cat([ortho6ds_id, ortho6ds], dim=-2)
        all_rots = rotation_6d_to_matrix(ortho6ds)

        r, t = multi_rigid(default_rot, default_trans, all_rots, torch.zeros_like(default_trans))

        r_chi1, t_chi1 = r[..., 4, :, :], t[..., 4, :]
        r_chi2, t_chi2 = multi_rigid(r_chi1, t_chi1, r[..., 5, :, :], t[..., 5, :])
        r_chi3, t_chi3 = multi_rigid(r_chi2, t_chi2, r[..., 6, :, :], t[..., 6, :])
        r_chi4, t_chi4 = multi_rigid(r_chi3, t_chi3, r[..., 7, :, :], t[..., 7, :])

        r = torch.cat([
            r[..., :5, :, :], r_chi2[..., None, :, :], r_chi3[..., None, :, :], r_chi4[..., None, :, :]
        ], dim=-3)
        t = torch.cat([
            t[..., :5, :], t_chi2[..., None, :], t_chi3[..., None, :], t_chi4[..., None, :]
        ], dim=-2)
        return r, t

    def local_frame8_to_local_atom37_torch(self, aatype: torch.Tensor, r: torch.Tensor, t: torch.Tensor):
        '''
        aatype: torch.tensor[B,L]
        r: torch.tensor[B,L,8,3,3]
        t: torch.tensor[B,L,8,3]
        '''
        residx_to_group_idx = self.restype_atom37_to_rigid_group_torch[aatype]  # [B, L, 37]
        lit_positions = self.restype_atom37_rigid_group_positions_torch[aatype]  # [B, L, 37, 3]
        r2 = torch.gather(r, index=residx_to_group_idx[..., None, None].expand([-1] * (len(aatype.shape) + 1) + [3, 3]),
                          dim=-3)
        t2 = torch.gather(t, index=residx_to_group_idx[..., None].expand([-1] * (len(aatype.shape) + 1) + [3]), dim=-2)
        r2 = r2.transpose(-1, -2)  # needs transpose
        out = apply_rigid(lit_positions, r2, t2)
        return out

    def torsion7_to_local_atom37_torch(self, aatype: torch.Tensor, sin_cos: torch.Tensor):
        '''
        aatype: torch.tensor[B,L]
        sin_cos: torch.tensor[B,L,7,2]
        '''
        r, t = self.torsion7_to_frame8_torch(aatype, sin_cos)
        local_positions = self.local_frame8_to_local_atom37_torch(aatype, r, t)
        return local_positions, t

    def ortho6ds_to_local_atom37_torch(self, aatype: torch.Tensor, ortho6ds: torch.Tensor):
        r, t = self.ortho6ds_to_frame8_torch(aatype, ortho6ds)
        local_positions = self.local_frame8_to_local_atom37_torch(aatype, r, t)
        return local_positions, t

    def atom37_to_atom14_torch(self, aatype: torch.Tensor, all_atom_pos: torch.Tensor):
        '''
        aatype: torch.tensor[B,L]
        all_atom_pos: torch.tensor[B,L,37,3]
        all_atom_pos: torch.tensor[B,L,37]
        '''
        device = aatype.device
        restype_atom37_to_atom14_index_torch = self.restype_atom37_to_atom14_index_torch.to(device)
        atom_indices = restype_atom37_to_atom14_index_torch[aatype]  # [B, L, 14]
        indices_temp = atom_indices[..., None].expand(
            [-1] * (len(aatype.shape) + 1) + [3])
        all_atom14_pos = torch.gather(all_atom_pos, index=indices_temp, dim=-2)
        return all_atom14_pos

    def atom37_to_atom14_mask_torch(self, aatype: torch.Tensor, all_atom_mask: torch.Tensor):
        device = aatype.device
        restype_atom37_to_atom14_index_torch = self.restype_atom37_to_atom14_index_torch.to(device)  # [B, L, 14]
        atom_indices = restype_atom37_to_atom14_index_torch[aatype]  # [B, L, 14]
        all_atom14_mask = torch.gather(all_atom_mask, index=atom_indices, dim=-1)
        return all_atom14_mask


transformer = TorsionPositionTransformer()
atom37_to_torsion7_torch = transformer.atom37_to_torsion7_torch
torsion7_to_local_atom37_torch = transformer.torsion7_to_local_atom37_torch
atom37_to_atom14_torch = transformer.atom37_to_atom14_torch
atom37_to_atom14_mask_torch = transformer.atom37_to_atom14_mask_torch
