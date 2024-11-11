import os
import json
import argparse
import torch
import numpy as np
import open3d as o3d

import edgegaussians.edge_extraction.fitting as fitting
import edgegaussians.edge_extraction.clustering as clustering
import edgegaussians.edge_extraction.filtering as filtering
import edgegaussians.utils.train_utils as train_utils
import edgegaussians.utils.io_utils as io_utils
import edgegaussians.utils.misc_utils as misc_utils
import edgegaussians.utils.eval_utils as eval_utils

from plyfile import PlyData

def filter_points(pos, scales, quats, opacities, filtering_conf, data_config, scene_name):

    if filtering_conf['filter_stat_outliers']:
        inlier_inds_ = filtering.filter_stat_outliers(pos, num_nn = filtering_conf["filter_stat_outlier_num_nn"], std_multiplier=filtering_conf["filter_stat_outlier_std_mult"])
        pos = pos[inlier_inds_]; scales = scales[inlier_inds_]; quats = quats[inlier_inds_]; opacities = opacities[inlier_inds_]

    if filtering_conf['filter_by_opacity']:
        inlier_inds_ = filtering.filter_by_opacity(opacities, min_opacity = filtering_conf['filter_opacity_min'])
        pos = pos[inlier_inds_]; scales = scales[inlier_inds_]; quats = quats[inlier_inds_]; opacities = opacities[inlier_inds_]
    
    if filtering_conf['filter_by_projection']:
        dataparser, images_dir, _ = train_utils.parse_data(data_config, scene_name)
        parser_type = data_config["parser_type"]
        image_res_scaling_factor = data_config["image_res_scaling_factor"]
        if_scale_scene_unit = data_config["scale_scene_unit"]

        # This needs to change if the scene is being scaled, right now it is assumed to be not be scaled
        _ = train_utils.init_views_and_get_scale(dataparser, 
                                images_dir, 
                                parser_type = parser_type,
                                image_res_scaling_factor = image_res_scaling_factor,
                                if_scale_scene_unit = False,
                                points_extent = None)
        
        edges, cameras = filtering.load_images_and_cameras(dataparser)
        inlier_inds_ = filtering.filter_by_projection(pos, edges, cameras)
        pos = pos[inlier_inds_]; scales = scales[inlier_inds_]; quats = quats[inlier_inds_]; opacities = opacities[inlier_inds_]
    
    return pos, scales, quats, opacities

def main():

    parser = argparse.ArgumentParser(description='Fit edges to the edge gaussians model')
    parser.add_argument('--config_file', type=str, help='Path to the configuration file')
    parser.add_argument('--scene_name', type=str, help='Name of the scene', default = None)
    parser.add_argument('--input_ply', type=str, help='Path to the ply file with the edge gaussians model')
    parser.add_argument('--save_filtered', action='store_true', help='Save the filtered points')
    parser.add_argument('--output_json', type=str, help='Path to the ply file with the fitted edges')
    parser.add_argument('--visualize_clusters', action='store_true', help='Visualize the clusters')
    parser.add_argument('--visualize_fit_edges', action='store_true', help='Visualize the fitted edges')
    parser.add_argument('--save_sampled_points', action='store_true', help='Save the sampled points')
    parser.add_argument('--sample_resolution', type=int, default=0.005, help='Resolution of the sampled points')
    

    args = parser.parse_args()
    assert os.path.exists(args.config_file), 'Configuration file does not exist'
    conf = json.load(open(args.config_file))
    
    if args.input_ply is None:
        output_config = conf['output']
        data_config = conf['data']
        edge_detector = data_config["edge_detection_method"]
        output_base = output_config["output_dir"]
        exp_name = output_config["exp_name"] + "_" + edge_detector
        output_dir = os.path.join(output_base, exp_name, args.scene_name)
        input_ply_path = os.path.join(output_dir, 'gaussians_all.ply')
    else:
        input_ply_path = args.input_ply
    
    if args.output_json is None:
        output_base = output_config["output_dir"]
        exp_name = output_config["exp_name"] + "_" + edge_detector
        output_dir = os.path.join(output_base, exp_name, args.scene_name)
        output_json = os.path.join(output_dir, 'parametric_edges.json')
    else:
        output_json = args.output_json

    data_config = conf['data']
    filtering_conf = conf['filtering']
    fitting_conf = conf['parametric_fitting']
    
    angle_thresh = fitting_conf['angle_thresh']
    line_ransac_thresh = fitting_conf['line_ransac_thresh']
    line_curve_residual_comp_factor = fitting_conf['line_curve_residual_comp_factor']

    min_cluster_size = fitting_conf['min_cluster_size']

    # Read the ply file exported after training the edge gaussians model
    pos, scales, quats, opacities = io_utils.read_gaussian_params_from_ply(input_ply_path)

    # Filter the points
    pos, scales, quats, opacities = filter_points(pos, scales, quats, opacities, filtering_conf, data_config, args.scene_name)

    major_dirs = misc_utils.get_major_directions_from_scales_quats(scales, quats)
    if args.save_filtered:
        # save a file with all params
        io_utils.write_gaussian_params_as_ply(pos, scales, quats, opacities, os.path.join(output_dir, 'gaussians_filtered.ply'))
        
        # save pts with major dirs for easy visualization
        io_utils.write_pts_with_major_dirs_as_ply(pos, major_dirs, os.path.join(output_dir, 'pts_with_major_dirs.ply')) 

    # cluster the points
    valid_clusters, points, directions = clustering.cluster_points_using_directions_greedy(pos,
                                                             major_dirs,
                                                             angle_thresh=angle_thresh,
                                                             visualize_clusters=args.visualize_clusters, 
                                                             min_cluster_size=min_cluster_size)
    
    print("Clustering complete")
    print(f"Number of clusters: {len(valid_clusters)}")

    # fit appropriate edges to the clusters
    _, parametric_edges_dict = fitting.fit_edges(valid_clusters,
                              pos,
                              major_dirs, 
                              ransac_thresh = line_ransac_thresh,
                              line_curve_residual_comp_factor = line_curve_residual_comp_factor,
                              visualize_fit_edges = False,
                              output_json = output_json
                              )

    # These can be used for evaluation
    if args.save_sampled_points:
        all_curve_points, all_line_points, _, _ = eval_utils.get_pred_points_and_directions_from_dict(parametric_edges_dict, sample_resolution = args.sample_resolution)
        pts = np.concatenate([all_curve_points, all_line_points], axis=0)
        if pts.shape[0] == 0:
            raise Exception("No points found")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0, 0, 0])
        o3d.io.write_point_cloud(os.path.join(output_dir, f'edge_sampled_points_{args.sample_resolution}.ply'), pcd)


if __name__ == '__main__':
    main()
