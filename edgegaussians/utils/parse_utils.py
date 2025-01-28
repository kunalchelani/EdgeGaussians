import json
import os

from pathlib import Path

from edgegaussians.data.dataparsers import DataParserFactory

def get_configs(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    output_config = config['output']

    return model_config, training_config, data_config, output_config


def get_paths_from_data_config(data_config, scene_name):
    
    if data_config["parser_type"] == "emap":
        base_dir = data_config['base_dir']
        edge_detection_method = data_config['edge_detection_method']
        
        # Works for ABC-NEF and replica at the moment

        data_dir = Path(base_dir) / scene_name  
        cameras_path = data_dir / "meta_data.json"
        images_dir = data_dir /  f"edge_{edge_detection_method}"
        if data_config["dataset_name"] in ["ABC", "Replica", "tnt"]:
            seed_points_ply_path = data_dir / "colmap/sparse/sparse.ply"
        elif data_config["dataset_name"] == "DTU":
            seed_points_ply_path = data_dir / "sparse_sfm_points.txt"
    
        # return strings
        return images_dir.as_posix(), cameras_path.as_posix(), seed_points_ply_path.as_posix()

    elif data_config["parser_type"] == "colmap":
        
        data_dir = Path(data_config['base_dir']) / scene_name
        images_dir = data_dir / f"edge_{data_config['edge_detection_method']}"
        images_dir.as_posix()
        
        colmap_base_dir = data_dir / "colmap"
        colmap_base_dir_str = colmap_base_dir.as_posix()

        sparse_points_path = colmap_base_dir / "sparse.ply"
        if not os.path.exists(sparse_points_path):
            sparse_points_path = colmap_base_dir / "points3D.bin"

        if not os.path.exists(sparse_points_path):
            sparse_points_path = colmap_base_dir / "points3D.txt"
        
        if not os.path.exists(sparse_points_path):
            sparse_points_path = None

        if sparse_points_path is not None:
            sparse_points_path_str = sparse_points_path.as_posix()
        else:
            sparse_points_path_str = None
            
        return images_dir.as_posix(), colmap_base_dir_str, sparse_points_path_str
    

def parse_data(data_config, scene_name):
    images_dir, data_input_path, seed_points_path = get_paths_from_data_config(data_config, scene_name)
    print(data_config)

    if data_config["parser_type"] == "colmap":
        print(f"Images dir : {images_dir}")
        print(f"Colmap base dir : {data_input_path}")
        print(f"Seed points path : {seed_points_path}")
        dp_kwargs = {"new_extension" : data_config["new_extension"]}

    elif data_config["parser_type"] == "emap":
        print(f"Images dir : {images_dir}")
        print(f"Cameras path : {data_input_path}")
        print(f"Seed points path : {seed_points_path}")
        dp_kwargs = {}
    
    dataparser = DataParserFactory.get_parser(data_config["parser_type"],
                                                   data_input_path,
                                                   dp_kwargs)
    return dataparser, images_dir, seed_points_path