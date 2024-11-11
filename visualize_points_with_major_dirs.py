import argparse
import numpy as np

from plyfile import PlyData, PlyElement

from edgegaussians.vis.vis_utils import visualize_points_with_major_dirs

def main():
    parser = argparse.ArgumentParser(description='Visualize points with major directions')
    parser.add_argument('--input_ply', type=str, help='Path to the ply file with the edge gaussians model')
    parser.add_argument('--vis_type', type=str, default='show')
    parser.add_argument('--line_scale', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    ply_path = args.input_ply

    vertex_data = PlyData.read(ply_path)['vertex']
    points_3d = np.hstack((vertex_data['x'][:, np.newaxis], vertex_data['y'][:, np.newaxis], vertex_data['z'][:, np.newaxis]))
    major_dirs = np.hstack((vertex_data['dir_x'][:, np.newaxis], vertex_data['dir_y'][:, np.newaxis], vertex_data['dir_z'][:, np.newaxis]))

    if args.vis_type == 'save':
        assert args.save_path is not None, 'Output ply path must be provided'

    visualize_points_with_major_dirs(points_3d, major_dirs, line_scale = args.line_scale, vis_type = args.vis_type, save_path = args.save_path)

if __name__ == '__main__':
    main()