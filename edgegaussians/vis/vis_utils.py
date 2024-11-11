import os
import open3d as o3d
import numpy as np
import ipdb

from edgegaussians.utils.colmap_read_write_model import qvec2rotmat
from edgegaussians.utils.misc_utils import get_major_directions_from_scales_quats

def visualize_clusters(points, clusters):
    colors = np.random.rand(len(clusters), 3)
    color_coded_points = o3d.geometry.PointCloud() 
    color_coded_points.points = o3d.utility.Vector3dVector(points)
    colors_pcd = np.ones((len(points), 3))
    for i, cluster in enumerate(clusters):
        cluster_inds = list(cluster)
        try:
            colors_pcd[cluster_inds] = colors[i]
        except:
            ipdb.set_trace()
    color_coded_points.colors = o3d.utility.Vector3dVector(colors_pcd)
    o3d.visualization.draw_geometries([color_coded_points])

def visualize_fit_edges(pts_lines, pts_curves, vis_method = 'sampled_points'):

    if vis_method == 'sampled_points':
        o3d_geometries = []

        for i, line in enumerate(pts_lines):
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line)
            line_pcd.paint_uniform_color([1, 0, 0])
            o3d_geometries.append(line_pcd)
        
        for i, curve in enumerate(pts_curves):
            curve_pcd = o3d.geometry.PointCloud()
            curve_pcd.points = o3d.utility.Vector3dVector(curve)
            curve_pcd.paint_uniform_color([0, 1, 0])
            o3d_geometries.append(curve_pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geom in o3d_geometries:
            vis.add_geometry(geom)
        vis.run()
        vis.destroy_window()
    
    # The line thickness can't be controlled in the current open3d version and hence this method
    # is less preferable, but this would be a better visualization method, I think. 
    elif vis_method == 'line_set':
        o3d_geometries = []
        for i, line in enumerate(pts_lines):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line)
            connections = np.array([[j, j+1] for j in range(len(line)-1)])
            line_set.lines = o3d.utility.Vector2iVector(connections)
            # color all lines red (1,0,0)
            line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (len(line)-1, 1)))
            o3d_geometries.append(line_set)

        for i, curve in enumerate(pts_curves):
            curve_lineset = o3d.geometry.LineSet()
            curve_lineset.points = o3d.utility.Vector3dVector(curve)
            curve_lineset.lines = o3d.utility.Vector2iVector(np.array([[j, j+1] for j in range(curve.shape[0]-1)]))
            # color all lines green (0,1,0)
            curve_lineset.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 1, 0]), (len(curve)-1, 1)))
            o3d_geometries.append(curve_lineset)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geom in o3d_geometries:
            vis.add_geometry(geom)
        vis.run()
        vis.destroy_window()

def visualize_points_with_major_dirs(points_3d, major_dirs, line_scale = 0.01, vis_type = "show", save_path = None):
    num_pts = points_3d.shape[0]

    p1 = points_3d + line_scale * major_dirs
    p2 = points_3d - line_scale * major_dirs

    lines = np.stack([np.arange(num_pts), np.arange(num_pts) + num_pts], axis=1)
        
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points_3d)
    o3d_pc.paint_uniform_color([0.2, 0.2, 0.2])

    o3d_lineset = o3d.geometry.LineSet()
    o3d_lineset.points = o3d.utility.Vector3dVector(np.vstack([p1, p2]))    
    lines = np.vstack(lines)
    o3d_lineset.lines = o3d.utility.Vector2iVector(lines)
    colors = [[1, 0, 0] for i in range(len(lines))]
    colors = np.array(colors)

    o3d_lineset.colors = o3d.utility.Vector3dVector(colors)
    
    if vis_type == "show":
        o3d.visualization.draw_geometries([o3d_lineset, o3d_pc])
    elif vis_type == "save":
        o3d.io.write_line_set(save_path, o3d_lineset)
    else:
        raise ValueError("Invalid vis_type")
        

        
def vis_colmap_cameras_open3d_lineset(images, cameras):
    o3d_lineset = o3d.geometry.LineSet()
    lineset_points = []
    connections = []
    colors = []
    # iterate over the cameras
    for i, imid in enumerate(images):
        image = images[imid]
        tvec = image.tvec
        qvec = image.qvec # this is in w,x,y,z format
        camera_id = image.camera_id
        colmap_camera = cameras[camera_id]
        width = colmap_camera.width
        height = colmap_camera.height
        params = colmap_camera.params
        
        # each camera would have 5 points - one at the camera and 4 at corners of the principal plane
        rotmat = qvec2rotmat(qvec)
        camera_center = -rotmat.T @ tvec
        z_axis = rotmat[:,2]
        x_axis = rotmat[:,0]
        y_axis = rotmat[:,1]
        ls = 0.05
        coord_sys_c = np.array([[0,0,0], [ls,0,0], [0,ls,0], [0,0,ls]], dtype=np.float32)
        coord_sys_w = rotmat.T @ (coord_sys_c.T - tvec.reshape(-1,1))
        lineset_points.extend([coord_sys_w[:,0], coord_sys_w[:,1], coord_sys_w[:,2], coord_sys_w[:,3]])
        connections.extend([[4*i, 4*i+j] for j in range(1,4)])
        colors.extend([[0,0,1], [0,1,0], [1,0,0]])
    
    lineset_points = np.array(lineset_points, dtype=np.float32)
    connections = np.array(connections, dtype=np.int32)
    colors = np.array(colors, dtype=np.float32)
    # ipdb.set_trace()
    o3d_lineset.points = o3d.utility.Vector3dVector(lineset_points)
    o3d_lineset.lines = o3d.utility.Vector2iVector(connections)
    o3d_lineset.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([o3d_lineset])
        
def vis_views_as_open3d_lineset(views):
    o3d_lineset = o3d.geometry.LineSet()
    lineset_points = []
    connections = []
    colors = []
    # iterate over the cameras
    for i, view in enumerate(views):
        camera = view['camera']
        rotmat = camera.R
        trans = camera.trans.numpy().reshape(-1,1)
        ls = 0.05        
        coord_sys_c = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32) * ls
        coord_sys_w = rotmat.T @ (coord_sys_c.T - trans.reshape(-1,1))
        lineset_points.extend([coord_sys_w[:,0], coord_sys_w[:,1], coord_sys_w[:,2], coord_sys_w[:,3]])
        connections.extend([[4*i, 4*i+j] for j in range(1,4)])
        colors.extend([[0,0,1], [0,1,0], [1,0,0]])
    
    lineset_points = np.array(lineset_points, dtype=np.float32)
    connections = np.array(connections, dtype=np.int32)
    colors = np.array(colors, dtype=np.float32)
    o3d_lineset.points = o3d.utility.Vector3dVector(lineset_points)
    o3d_lineset.lines = o3d.utility.Vector2iVector(connections)
    o3d_lineset.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([o3d_lineset])