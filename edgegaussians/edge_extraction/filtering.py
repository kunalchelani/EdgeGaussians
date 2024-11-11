import os
import json
import numpy as np
import open3d as o3d
import cv2
import ipdb

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def load_from_json(filename):
    """Load a dictionary from a JSON filename."""
    assert filename.split(".")[-1] == "json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def project2D_single(K, R, T, points3d):
    shape = points3d.shape
    assert shape[-1] == 3
    X = points3d.reshape(-1, 3)

    x = K @ (R @ X.T + T)
    x = x.T
    x = x / x[:, -1:]
    x = x.reshape(*shape)[..., :2].reshape(-1, 2).tolist()
    return x

def project2D(K, R, T, all_curve_points, all_line_points):
    all_curve_uv, all_line_uv = [], []
    for curve_points in all_curve_points:
        curve_points = np.array(curve_points).reshape(-1, 3)
        curve_uv = project2D_single(K, R, T, curve_points)
        all_curve_uv.append(curve_uv)
    for line_points in all_line_points:
        line_points = np.array(line_points).reshape(-1, 3)
        line_uv = project2D_single(K, R, T, line_points)
        all_line_uv.append(line_uv)
    return all_curve_uv, all_line_uv

def load_images_and_cameras(dataparser):

    views = [view['camera'] for view in dataparser.views]
    edge_images = [view['image']/255.0 for view in dataparser.views]
    cameras = [None for _ in range(len(views))]
    h, w = views[0].height, views[0].width

    for i in range(len(views)):
        K = views[i].get_K().cpu().numpy()
        viewmat = views[i].viewmat.cpu().numpy()
        R = viewmat[:3, :3]
        t = viewmat[:3, 3:]
        cameras[i] = {'K' : K, 'R' : R, 't' : t, 'h' : h, 'w' : w}
    
    return edge_images, cameras


def filter_stat_outliers(means : np.ndarray, num_nn : int = 10, std_multiplier: float = 3.0):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)
    print("Statistical oulier removal")
    cl, inlier_inds = pcd.remove_statistical_outlier(nb_neighbors=num_nn,
                                                        std_ratio=std_multiplier)
    
    print(f"Removed {len(means) - len(inlier_inds)} outliers")

    return np.array(inlier_inds).reshape(-1)

def filter_by_opacity(opacities : np.ndarray, min_opacity : float):
    num_pts = opacities.shape[0]
    print(f"Num points before filtering by opacity: {num_pts}")
    inlier_inds =  opacities > min_opacity
    print(f"Removed {len(opacities) - np.sum(inlier_inds)} points")

    return inlier_inds.reshape(-1)


def filter_by_projection(gaussian_means,
                         edge_images,
                         cameras,
                         visib_thresh:float = 0.1):

    num_gs = gaussian_means.shape[0]
    num_images = len(edge_images)
    print(f"Num points before filtering by projection: {num_gs}")
    gs_visib_matrix = np.zeros((num_gs, num_images))

    for i in range(num_images):
        
        K = cameras[i]['K']
        R = cameras[i]['R']
        t = cameras[i]['t']
        h = cameras[i]['h']
        w = cameras[i]['w']
        all_curve_uv, _ = project2D(
            K, R, t, [gaussian_means], []
        )
        edge_uv = all_curve_uv[0]
        edge_uv = np.array(edge_uv)
        if len(edge_uv) == 0:
            continue
        edge_uv = np.round(edge_uv).astype(np.int32)
        edge_u = edge_uv[:, 0]
        edge_v = edge_uv[:, 1]

        edge_map = edge_images[i].cpu().numpy()

        valid_mask = (edge_u >= 0) & (edge_u < w) & (edge_v >= 0) & (edge_v < h)
        valid_edge_uv = edge_uv[valid_mask,:]

        if len(valid_edge_uv) > 0:
            
            projected_edge = edge_map[valid_edge_uv[:, 1], valid_edge_uv[:, 0]]
            gs_visib_matrix[valid_mask, i] += projected_edge

    gs_visib = np.mean(gs_visib_matrix, axis=1)
    inlier_inds = gs_visib > visib_thresh

    print(f"Removed {num_gs - np.sum(inlier_inds)} points")

    return inlier_inds.reshape(-1)