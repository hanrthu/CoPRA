import torch
import numpy as np

from typing import Optional

from torch.nn import functional as F
ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4

def get_pos_CB(pos14, atom_mask):
    """
    Args:
        pos14:  (N, L, 14, 3)
        atom_mask:  (N, L, 14)
    """
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]  # (N, L)
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA]   # (N, L, 3)
    pos_CB = pos14[:, :, ATOM_CB]
    return torch.where(mask_CB, pos_CB, pos_CA)


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out

def pairwise_distances(x, y=None, return_v=False):
    """
    Args:
        x:  (B, N, d)
        y:  (B, M, d)
    """
    if y is None: y = x
    v = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, d)
    d = safe_norm(v, dim=-1)
    if return_v:
        return d, v
    else:
        return d

def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)



def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    if t is None:
        p = torch.matmul(R.transpose(-1, -2), q)  # (N, L, 3, *)
    else:
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1) )
    dihed = torch.nan_to_num(dihed)
    return dihed


def knn_gather(idx, value):
    """
    Args:
        idx:    (B, N, K)
        value:  (B, M, d)
    Returns:
        (B, N, K, d)
    """
    N, d = idx.size(1), value.size(-1)
    idx = idx.unsqueeze(-1).repeat(1, 1, 1, d)      # (B, N, K, d)
    value = value.unsqueeze(1).repeat(1, N, 1, 1)   # (B, N, M, d)
    return torch.gather(value, dim=2, index=idx)


def knn_points(q, p, K):
    """
    Args:
        q: (B, M, d)
        p: (B, N, d)
    Returns:
        (B, M, K), (B, M, K), (B, M, K, d)
    """
    _, L, _ = p.size()
    d = pairwise_distances(q, p)  # (B, N, M)
    dist, idx = d.topk(min(L, K), dim=-1, largest=False)  # (B, M, K), (B, M, K)
    return dist, idx, knn_gather(idx, p)


def angstrom_to_nm(x):
    return x / 10


def nm_to_angstrom(x):
    return x * 10


def get_backbone_dihedral_angles(pos_atoms, chain_nb, res_nb, mask):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
        chain_nb:   (N, L).
        res_nb:     (N, L).
        mask:       (N, L).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
        mask_bb_dihed:  Masks of dihedral angles, (N, L, 3).
    """
    pos_N  = pos_atoms[:, :, 0]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, 1]
    pos_C  = pos_atoms[:, :, 2]

    N_term_flag, C_term_flag = get_terminus_flag(chain_nb, res_nb, mask)  # (N, L)
    omega_mask = torch.logical_not(N_term_flag)
    phi_mask = torch.logical_not(N_term_flag)
    psi_mask = torch.logical_not(C_term_flag)

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]),
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    mask_bb_dihed = torch.stack([omega_mask, phi_mask, psi_mask], dim=-1)
    bb_dihedral = torch.stack([omega, phi, psi], dim=-1) * mask_bb_dihed
    return bb_dihedral, mask_bb_dihed


def pairwise_dihedrals(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        Inter-residue Phi and Psi angles, (N, L, L, 2).
    """
    N, L = pos_atoms.shape[:2]
    pos_N  = pos_atoms[:, :, 0]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, 1]
    pos_C  = pos_atoms[:, :, 2]

    ir_phi = dihedral_from_four_points(
        pos_C[:,:,None].expand(N, L, L, 3),
        pos_N[:,None,:].expand(N, L, L, 3),
        pos_CA[:,None,:].expand(N, L, L, 3),
        pos_C[:,None,:].expand(N, L, L, 3)
    )
    ir_psi = dihedral_from_four_points(
        pos_N[:,:,None].expand(N, L, L, 3),
        pos_CA[:,:,None].expand(N, L, L, 3),
        pos_C[:,:,None].expand(N, L, L, 3),
        pos_N[:,None,:].expand(N, L, L, 3)
    )
    ir_dihed = torch.stack([ir_phi, ir_psi], dim=-1)
    return ir_dihed

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v * w, dim=-1)

    return torch.acos(vw)


def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor of shape [batch,nres]
          stores resulting dihedrals
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

    return torch.atan2(y, x)


def backbone_torsion(N, Ca, C, O):
    # N, Ca, C, O
    omega = get_dih(Ca[:-1], C[:-1], N[1:], Ca[1:])
    phi = get_dih(C[:-1], N[1:], Ca[1:], C[1:])
    psi = get_dih(N[:-1], Ca[:-1], C[:-1], O[:-1])

    omega = F.pad(omega, (0, 1))
    phi = F.pad(phi, (1, 0))
    psi = F.pad(psi, (0, 1))

    omega = omega % (2 * np.pi)
    phi = phi % (2 * np.pi)
    pspsi_af2i = psi % (2 * np.pi)

    return omega, phi, psi


def get_rotation(p1, o, p2):
    '''
    - p1, o, p2 (N, Ca, C)

    Returns:
    --------------------
    r: np.array (3,3)
    '''
    v1 = p1 - o
    v2 = p2 - o
    e1 = v1 / np.linalg.norm(v1 + 1e-12, ord=2)
    e2 = v2 - v2.dot(e1) * e1
    e2 = e2 / np.linalg.norm(e2 + 1e-12, ord=2)
    e3 = np.cross(e1, e2)
    r = np.stack([e1, e2, e3]).T
    return r


def get_batch_rotation(p1, o, p2):
    '''
    - p1, o, p2 (N, Ca, C)

    Args:
    --------------------
    p1: np.array(N,3)
    o: np.array(N,3)
    p2: np.array(N,3)

    Returns:
    --------------------
    r: np.array(N,3,3)
    '''
    v1 = p1 - o
    v2 = p2 - o
    e1 = v1 / np.linalg.norm(v1 + 1e-12, ord=2, keepdims=True, axis=-1)
    e2 = v2 - (v2 * e1).sum(axis=-1, keepdims=True) * e1
    e3 = np.cross(e1, e2)
    e = np.stack([e1, e2, e3]).transpose(1, 2, 0)  # (N,3,3)
    return e


def normalize_coord_by_first_res(coords_norm, idx_ca=1, idx_c=2):
    Ca0 = coords_norm[0, idx_ca]
    C = coords_norm[0, idx_c]
    Ca1 = coords_norm[1, idx_ca]
    r = get_rotation(Ca1, Ca0, C)
    if np.isnan(r).any():
        r = np.eye(3)
    t = Ca0
    coords_norm = coords_norm - Ca0
    coords_norm = coords_norm @ r
    return coords_norm


def get_local_rotatation(src, dst):
    '''
    src@r_t -> dst
    '''
    zeros = torch.zeros_like(src)
    angles = get_ang(src, zeros, dst)

    n = torch.cross(src, dst)
    n = F.normalize(n, p=2, dim=-1)

    axis_angle = angles[..., None] * -n
    r_t = axis_angle_to_matrix(axis_angle).float()
    is_collinear = torch.logical_or(torch.isnan(angles), torch.abs(angles) < 1e-4)
    is_collinear = is_collinear[..., None, None].expand_as(r_t)
    eye = torch.eye(3, device=r_t.device).unsqueeze(0).expand_as(r_t)

    r_t = torch.where(is_collinear, eye, r_t)
    return r_t


def apply_rigid(x: torch.Tensor, rotation: torch.Tensor, translation: Optional[torch.Tensor] = None):
    '''
    x: (...,3)
    rotation: (...,3,3)
    translation: (...,3)

    Outputs:
    --------------------
    out: (...,3)
    '''
    if translation is None:
        translation = torch.zeros_like(x)
    assert len(x.shape) == len(translation.shape)
    x = x[..., None, :]
    translation = translation[..., None, :]
    out = x @ rotation + translation
    out = out.squeeze(-2)
    return out


def inv_rigid(rotation, translation):
    '''
    rotation: (...,N,3,3)
    translation: (...,N,3)

    Outputs:
    --------------------
    rotation_new: (...,N,3,3)
    translation_new: (...,N,3)
    '''
    translation = translation[..., None, :]
    rotation_new = rotation.transpose(-1, -2)
    translation_new = -translation @ rotation_new
    translation_new = translation_new.squeeze(-2)

    return rotation_new, translation_new


def get_frame_from_coords(coords):
    '''
    return r_global_to_local, t_global_to_local
    '''
    N = coords[:, 0]  # 'N', 'CA', 'C', 'CB', 'O', 'CG'
    Ca = coords[:, 1]
    C = coords[:, 2]
    if not torch.is_tensor(N):
        N = torch.tensor(N)
        Ca = torch.tensor(Ca)
        C = torch.tensor(C)
    frames_rotation = get_batch_rotation_torch(C, Ca, N)  # (L, 3, 3)
    frames_rotation = frames_rotation.transpose(-1, -2)  # local_to_global
    frames_translation = Ca  # (L, 3)
    r_global_to_local, t_global_to_local = inv_rigid(frames_rotation, frames_translation)

    return r_global_to_local, t_global_to_local


def multi_rigid(rotation1, translation1, rotation2, translation2):
    '''
    x: (...,3)
    rotation: (...,3,3)
    translation: (...,3)

    Outputs:
    --------------------
    out: (...,3)
    '''
    translation1 = translation1[..., :, None]
    translation2 = translation2[..., :, None]
    translation = rotation1 @ translation2 + translation1
    rotation = rotation1 @ rotation2
    translation = translation.squeeze(-1)
    return rotation, translation


def get_batch_rotation_torch(p1, o, p2):
    '''
    - p1, o, p2 (N, Ca, C)
    - T(local->global)

    Args:
    --------------------
    p1: np.array(N,3)
    o: np.array(N,3)
    p2: np.array(N,3)

    Returns:
    --------------------
    r: np.array(N,3,3)
    '''
    v1 = p1 - o
    v2 = p2 - o
    e1 = F.normalize(v1 + 1e-12, p=2, dim=-1)
    e2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2 + 1e-12, p=2, dim=-1)
    e3 = torch.cross(e1, e2)
    e = torch.stack([e1, e2, e3]).permute(1, 2, 0)  # (N,3,3)
    return e


def get_frame_from_coords_batch(coords):
    b = coords.shape[0]
    rs = []
    ts = []
    for i in range(b):
        r, t = get_frame_from_coords(coords[i])
        rs += [r]
        ts += [t]
    r_global_to_local = torch.stack(rs)
    t_global_to_local = torch.stack(ts)
    return r_global_to_local, t_global_to_local
