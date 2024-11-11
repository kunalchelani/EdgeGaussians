import ipdb
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import json

from skimage.measure import LineModelND, ransac
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit

import edgegaussians.vis.vis_utils as vis_utils

##### Taken from EMAP #####

def bezier_curve(tt, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
    n = len(tt)
    matrix_t = np.concatenate(
        [(tt**3)[..., None], (tt**2)[..., None], tt[..., None], np.ones((n, 1))],
        axis=1,
    ).astype(float)
    matrix_w = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    ).astype(float)
    matrix_p = np.array(
        [[p0, p1, p2], [p3, p4, p5], [p6, p7, p8], [p9, p10, p11]]
    ).astype(float)
    return np.dot(np.dot(matrix_t, matrix_w), matrix_p).reshape(-1)


def line_fitting(endpoints):
    center = np.mean(endpoints, axis=0)

    # compute the main direction through SVD
    endpoints_centered = endpoints - center
    u, s, vh = np.linalg.svd(endpoints_centered, full_matrices=False)
    lamda = s[0] / np.sum(s)
    main_direction = vh[0]
    main_direction = main_direction / np.linalg.norm(main_direction)

    # project endpoints onto the main direction
    projections = []
    for endpoint_centered in endpoints_centered:
        projections.append(np.dot(endpoint_centered, main_direction))
    projections = np.array(projections)

    # construct final line
    straight_line = np.zeros(6)
    # print(np.min(projections), np.max(projections))
    straight_line[:3] = center + main_direction * np.min(projections)
    straight_line[3:] = center + main_direction * np.max(projections)

    return straight_line, lamda

def bezier_fit2(xyz, error_threshold=1.0):
    n = len(xyz)
    t = np.linspace(0, 1, n)
    xyz = xyz.reshape(-1)

    popt, _ = curve_fit(bezier_curve, t, xyz)

    # Generate fitted curve
    fitted_curve = bezier_curve(t, *popt).reshape(-1, 3)

    # Calculate residuals
    residuals = xyz.reshape(-1, 3) - fitted_curve

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))

    if rmse > error_threshold:
        return None
    else:
        return popt, residuals, fitted_curve


#### Simple fitting ours #####

def fit_edges(clusters, pts, dirs, 
                  ransac_thresh: float =  0.005,
                  line_curve_residual_comp_factor: float = 0.25,
                  visualize_fit_edges: bool = False,
                  output_json: str = None):
    
    curve_points = []
    eps_all = []
    conns = []

    edges = []
    all_line_pts = []
    for i,cluster in enumerate(clusters):
        eps = []
        try:
            pts_curr = pts[list(cluster)]
            _, inliers = ransac(pts_curr, LineModelND, min_samples=2,
                                    residual_threshold=ransac_thresh, max_trials=1000)
            
            
            line = pts_curr[inliers]
            line_eps, _ = line_fitting(line)
            main_direction = line_eps[3:] - line_eps[:3]
            main_direction /= np.linalg.norm(main_direction)
            
            # by projecting the points onto the line, and using the mean as center
            # we can get a sorting from one end to another

            mean_pt = (line_eps[3:] + line_eps[:3])/2

            lines_to_point = (pts_curr - mean_pt)
            dirs_to_point = lines_to_point / np.linalg.norm(lines_to_point, axis=1)[:, np.newaxis]
            normals = np.cross(main_direction, dirs_to_point)
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
            normals2 = np.cross(main_direction, normals)
            normals2 /= np.linalg.norm(normals2, axis=1)[:, np.newaxis]
        
            lamdas = np.dot(lines_to_point, main_direction)
            residuals_line_fit = np.abs(np.sum(np.multiply(normals2, lines_to_point), axis=1))
            mean_residual_line = np.mean(residuals_line_fit)
            lamda_order = np.argsort(lamdas)
            lamdas_sorted = lamdas[lamda_order]
            pts_curr = pts_curr[lamda_order]
            
            # now fit a bezier curve through the points and see if the residuals change quite a bit
            # if they do, then this is probably a curve
            
            out = bezier_fit2(pts_curr)
            if out is not None:
                popt, residuals, _ = out
                t_fit = (lamdas_sorted - np.min(lamdas_sorted)) / (np.max(lamdas_sorted) - np.min(lamdas_sorted))
                fitted_curve = bezier_curve(t_fit, *popt).reshape(-1, 3)
                fitted_curve_dense = bezier_curve(np.linspace(0, 1, 1000), *popt).reshape(-1, 3)
                # residuals are the minumum distance from the points to the curve
                residuals = cdist(pts_curr, fitted_curve_dense, 'euclidean')
                residuals = np.min(residuals, axis=1)
                mean_residual_curve = np.mean(residuals)

                if mean_residual_curve < line_curve_residual_comp_factor * mean_residual_line:
                    # print(f"Fitting bezier curve through cluster {i}")
                    edges.append({"type" : "curve", "popt" : popt, "all_pts" : pts_curr})
                    curve_points.append(fitted_curve_dense)
                    continue
                else:
                    # sample 100 points between line_eps[:3] and line_eps[3:]

                    t_fit = np.linspace(0, 1, 100)
                    fitted_line = line_eps[:3] + t_fit[:, np.newaxis] * (line_eps[3:] - line_eps[:3])
                    all_line_pts.append(fitted_line.tolist())

            eps.append(line_eps[:3])
            eps.append(line_eps[3:])
            eps_all.append(line_eps[:3])
            eps_all.append(line_eps[3:])
            ind1, ind2 = 2*len(conns) , 2*len(conns) + 1
            conns.append([ind1, ind2])

        except:
            print(f"Failed to fit line through cluster {i}")

        edges.append({"type" : "line", "eps" : eps, "conns" : conns, "all_pts" : pts_curr})

    if visualize_fit_edges:
        vis_utils.visualize_fit_edges(all_line_pts, curve_points)

    if output_json is not None:
        parametric_edges_dict = {"curves_ctl_pts" : [], "lines_end_pts" : []}
        for edge in edges:
            if edge["type"] == "curve":
                ctl_pts_matrix = np.array(edge["popt"]).reshape(-1, 3)
                ctl_pts = [ctl_pts_matrix[i].tolist() for i in range(4)]
                parametric_edges_dict["curves_ctl_pts"].append(ctl_pts)
            else:
                eps = edge["eps"][0].tolist() + edge["eps"][1].tolist()
                parametric_edges_dict["lines_end_pts"].append(eps)

        with open(output_json, "w") as f:
            json.dump(parametric_edges_dict, f)

    return edges, parametric_edges_dict