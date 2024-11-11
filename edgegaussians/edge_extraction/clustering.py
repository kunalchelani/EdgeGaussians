import numpy as np
import ipdb
import torch

import open3d as o3d
import edgegaussians.vis.vis_utils as vis_utils

from sklearn.neighbors import NearestNeighbors

# This method picks a point and greedily tries to grow it in both directions.
# Possible improvement:
# If a point candidate does not align well (due to local noise in the position) with the current cluster direction, it is not added and prevents growth of the cluster in that direction.
# 1. This could be improved by merging clusters later on if they are close enough., or by making the clustering method robust

def cluster_points_using_directions_greedy(points, directions, 
                        angle_thresh: float = 0.65,
                        min_cluster_size: int = 5,
                        visualize_clusters: bool = False):

    
    nn_model = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean").fit(points)
    distances_nn, indices_nn = nn_model.kneighbors(points)
    distances_nn = distances_nn[:, 1:]
    indices_nn = indices_nn[:, 1:]

    unvisited_points = set(range(len(points)))
    clusters = []

    while len(unvisited_points) > 0:
        selected_point = np.random.choice(list(unvisited_points))
        current_cluster = set([selected_point])
        current_cluster_direction = directions[selected_point]

        while len(set(current_cluster).intersection(unvisited_points)) > 0:
            unvisited_points.remove(selected_point)
            init_direction = directions[selected_point]
            connected_current = [selected_point]

            # get the 5 nearest neighbors of the selected_point
            # for the neighbors of the selected point, get the three alignments, alignment of their directions, and the alignment of their directions with direction vector of the selected point
            dir_between_points = points[indices_nn[selected_point,:]] - points[selected_point]
            dir_between_points /= np.linalg.norm(dir_between_points, axis=1)[:, np.newaxis]
            dirs_at_points = directions[indices_nn[selected_point,:]]

            align_dirs_at_pts = np.abs(np.dot(dirs_at_points, directions[selected_point].T))
            align_dirs_between_pt_dir_at_curr = np.abs(np.dot(dir_between_points, directions[selected_point].T))
            align_dirs_between_pt_dir_at_nbr = np.diag(np.abs(np.dot(dir_between_points, directions[indices_nn[selected_point]].T)))
            align_with_curr_cluster = np.abs(np.dot(dirs_at_points, current_cluster_direction.T))
            
            valid_additions = align_dirs_at_pts > angle_thresh
            valid_additions &= align_dirs_between_pt_dir_at_curr > angle_thresh
            valid_additions &= align_dirs_between_pt_dir_at_nbr > angle_thresh
            valid_additions &= align_with_curr_cluster > angle_thresh

            # ipdb.set_trace()
            current_cluster = current_cluster.union(set(indices_nn[selected_point, valid_additions].tolist()))
            current_cluster_aligned_directions = np.array([directions[i] if np.dot(directions[i], init_direction) > 0 else -directions[i] for i in current_cluster])
            # could be buggy because of mean not working as expected
            current_cluster_direction = np.mean(current_cluster_aligned_directions, axis=0)

            # choose any of the unvisited points in the current cluster
            if len(set(current_cluster).intersection(unvisited_points)) > 0:
                selected_point = list(set(current_cluster).intersection(unvisited_points))[0]
        
        clusters.append(current_cluster)

    valid_clusters = [cluster for cluster in clusters if len(cluster) > min_cluster_size]

    if visualize_clusters:
        vis_utils.visualize_clusters(points, valid_clusters)
        
        
    return valid_clusters, points, directions