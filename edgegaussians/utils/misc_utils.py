import torch
import math
import numpy as np
import torch.nn.functional as F

from typing import Tuple

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

def quats_to_rotmats_tensor(quaternions):
    """
    Convert a tensor of quaternions to rotation matrices.
    :param quaternions: Tensor of shape (N, 4), where N is the number of quaternions.
    :return: Tensor of shape (N, 3, 3) containing the rotation matrices.
    """
    # Ensure quaternions are normalized
    if len(quaternions.shape) == 1:
        quaternions = quaternions.unsqueeze(0)

    quaternions = F.normalize(quaternions, p=2, dim=1)

    # Extract individual components
    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]

    # Compute the rotation matrices
    R = torch.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    
    R[:, 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    
    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    
    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)
    
    return R

def quats_to_rotmats_np(quaternions):
    """
    Convert a tensor of quaternions to rotation matrices.
    :param quaternions: Numpy arra of shape (N, 4), where N is the number of quaternions.
    :return: Numpy array of shape (N, 3, 3) containing the rotation matrices.
    """
    # Ensure quaternions are normalized
    if len(quaternions.shape) == 1:
        quaternions = quaternions.unsqueeze(0)

    quaternions = quaternions / np.linalg.norm(quaternions, axis=1)[:, np.newaxis]

    # Extract individual components
    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]

    # Compute the rotation matrices
    R = np.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype)
    
    R[:, 0, 0] = 1 - 2 * (q2 ** 2 + q3 ** 2)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)
    
    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1 ** 2 + q3 ** 2)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)
    
    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1 ** 2 + q2 ** 2)
    
    return R


def get_major_directions_from_scales_quats(scales : np.ndarray, quats: np.ndarray):
    # scales: (N, 3)
    # quats: (N, 4) with quaternions in wxyz format
    rotmats = quats_to_rotmats_np(quats)
    argmax_scales = np.argmax(scales, axis=1)
    major_dirs = rotmats[np.arange(scales.shape[0]), :, argmax_scales]
    return major_dirs