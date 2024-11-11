from plyfile import PlyData, PlyElement
import numpy as np

def write_gaussian_params_as_ply(means, scales, quats, opacities, ply_path):
    n_gaussians = means.shape[0]
    vertex = np.zeros(n_gaussians, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                        ('scale1', 'f4'), ('scale2', 'f4'), ('scale3', 'f4'),
                                        ('quat1', 'f4'), ('quat2', 'f4'), ('quat3', 'f4'), ('quat4', 'f4'),
                                        ('opacity', 'f4')])
    
    vertex['x'] = means[:, 0]
    vertex['y'] = means[:, 1]
    vertex['z'] = means[:, 2]
    vertex['scale1'] = scales[:, 0]
    vertex['scale2'] = scales[:, 1]
    vertex['scale3'] = scales[:, 2]
    vertex['quat1'] = quats[:, 0]
    vertex['quat2'] = quats[:, 1]
    vertex['quat3'] = quats[:, 2]
    vertex['quat4'] = quats[:, 3]
    vertex['opacity'] = opacities[:, 0]

    vertex_element = PlyElement.describe(vertex, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)



def read_gaussian_params_from_ply(ply_path):

    plydata = PlyData.read(ply_path)
    data = plydata['vertex']

    pos = np.hstack((data['x'][:, np.newaxis], data['y'][:, np.newaxis], data['z'][:, np.newaxis]))
    scales = np.hstack((data['scale1'][:, np.newaxis], data['scale2'][:, np.newaxis], data['scale3'][:, np.newaxis]))
    quats = np.hstack((data['quat1'][:, np.newaxis], data['quat2'][:, np.newaxis], data['quat3'][:, np.newaxis], data['quat4'][:, np.newaxis]))    
    opacities = data['opacity'][:, np.newaxis]

    return pos, scales, quats, opacities

def write_pts_with_major_dirs_as_ply(pos, dirs, ply_path):
    num_pts = pos.shape[0]
    vertex_with_dir = np.zeros(num_pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('dir_x', 'f4'), ('dir_y', 'f4'), ('dir_z', 'f4')])
    
    vertex_with_dir['x'] = pos[:, 0]
    vertex_with_dir['y'] = pos[:, 1]
    vertex_with_dir['z'] = pos[:, 2]
    vertex_with_dir['dir_x'] = dirs[:, 0]
    vertex_with_dir['dir_y'] = dirs[:, 1]
    vertex_with_dir['dir_z'] = dirs[:, 2]

    vertex_element = PlyElement.describe(vertex_with_dir, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

def read_pts_with_major_dirs_from_ply(file_path):
    
    plydata = PlyData.read(file_path)
    data = plydata['vertex']

    pos = np.hstack((data['x'][:, np.newaxis], data['y'][:, np.newaxis], data['z'][:, np.newaxis]))
    dirs = np.hstack((data['dir_x'][:, np.newaxis], data['dir_y'][:, np.newaxis], data['dir_z'][:, np.newaxis]))

    return pos, dirs