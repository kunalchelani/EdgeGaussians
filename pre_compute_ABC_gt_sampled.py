import os
import argparse

import open3d as o3d
import numpy as np

import edgegaussians.utils.eval_utils as eval_utils


def main():
    parser = argparse.ArgumentParser(description="evaluate the results")
    parser.add_argument("--data_dir", type=str, help="Path to the data directory")
    parser.add_argument("--resolution", type=float, default=0.005, help="Resolution of the sampled points")
    # data dir should have the the feat and obj folders, as structured by EMAP

    args = parser.parse_args()

    gt_base_dir = os.path.join(args.data_dir, "groundtruth")
    scan_names = os.listdir(os.path.join(args.data_dir, "data"))
    
    for scan_name in scan_names:
        print(f"Evaluating {scan_name}")

        gt_points = eval_utils.get_gt_points(scan_name,
                        edge_type="all",
                        interval=args.resolution,
                        return_direction=False,
                        data_base_dir=gt_base_dir)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points[1])
        pcd_gt.paint_uniform_color([0, 1, 0])
        save_path = os.path.join(args.data_dir, "groundtruth", "sampled_pts", f"{scan_name}_{args.resolution}.ply")
        o3d.io.write_point_cloud(save_path, pcd_gt)

if __name__ == "__main__":
    main() 