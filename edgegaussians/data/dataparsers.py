import numpy as np
import torch
import json

from PIL import Image
from pathlib import Path

from edgegaussians.utils.colmap_read_write_model import read_cameras_binary, read_cameras_text, read_images_binary, read_images_text
from edgegaussians.cameras.cameras import Camera, OpenCVCamera

class DataParser():

    def __init__(self,) -> None:
        pass

    def load_views(self, images_dir: str):
        pass

    def load_image(self, image_dir: str, image_name: str):
        
        # load all the files with provided extension in the images directory
        image_path = Path(image_dir) / image_name
        if not image_path.exists():
            if image_path.suffix in [".jpg",  ".JPG", "jpeg", ".JPEG"]:
                image_path = Path(image_dir) / (image_name.split(".")[0] + ".png")
                if not image_path.exists():
                    image_path = Path(image_dir) / (image_name.split(".")[0] + ".PNG")
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
        image = Image.open(image_path)
        # convert to pytorch tensor
        image = torch.tensor(np.array(image), dtype=torch.float32)

        return image
    

class ColmapDataParser(DataParser):

    def __init__(self, base_path:str, new_extension: str = None) -> None:
        super().__init__()
        self.base_path = Path(base_path) # The colmap base folder, should contain images.txt or images.bin
        self.colmap_images_file_path = self.base_path / "images.txt"
        if not self.colmap_images_file_path.exists():
            self.colmap_images_file_path = self.base_path / "images.bin"

        self.cameras_file_path = self.base_path / "cameras.txt"
        if not self.cameras_file_path.exists():
            self.cameras_file_path = self.base_path / "cameras.bin"
        self.new_extension = new_extension
        self.views = []

    def load_views(self, images_dir: str, 
                   image_res_scaling_factor: float = 1.0):

        # load the intrinsics first
        if self.cameras_file_path.suffix == ".txt":
            colmap_cameras = read_cameras_text(self.cameras_file_path)
        elif self.cameras_file_path.suffix == ".bin":
            colmap_cameras = read_cameras_binary(self.cameras_file_path)
        else:
            raise ValueError(f"Unsupported file format for cameras file: {self.cameras_file_path.suffix}")
        
        if self.colmap_images_file_path.suffix == ".txt":
            images = read_images_text(self.colmap_images_file_path)
        elif self.colmap_images_file_path.suffix == ".bin":
            images = read_images_binary(self.colmap_images_file_path)
        
        for imid in images:
            image = images[imid]
            tvec = image.tvec
            qvec = image.qvec # this is in w,x,y,z format
            camera_id = image.camera_id
            colmap_camera = colmap_cameras[camera_id]
            width = colmap_camera.width
            height = colmap_camera.height
            params = colmap_camera.params

            # only simple pinhole camera support for now
            assert (colmap_camera.model == "SIMPLE_PINHOLE") or (colmap_camera.model == "PINHOLE"), f"Model : {colmap_camera.model}. Only simple pinhole camera model is supported for now."
            camera = Camera(height, width, params[0], params[1], params[2], params[3], qvec, tvec, scaling_factor=image_res_scaling_factor)
            if self.new_extension is not None:
                print(image.name)
                # remove the part following the last dot
                image_name_split = image.name.split(".")
                image_name = ".".join(image_name_split[:-1]) + self.new_extension
                print(image_name)
            else:
                print("New extension not provided. Using the same extension as the original image.")
                image_name = image.name
            print(image.name)
            image = self.load_image(images_dir, image_name)
            self.views.append({"camera" : camera, "image" : image})


class EMAPDataParser(DataParser):

    def __init__(self, meta_file_path: str) -> None:
        super().__init__()
        self.meta_file_path = Path(meta_file_path)
        self.views = []
    
    def load_views(self, images_dir: str):
        # read the meta_data.json file
        with open(self.meta_file_path, "r") as f:
            meta_data = json.load(f)
            height = meta_data["height"]
            width = meta_data["width"]
            frames = meta_data["frames"]
        
        for frame in frames:
            imname = frame["rgb_path"]
            cam_to_world = np.array(frame["camtoworld"])
            Kmat = np.array(frame["intrinsics"])

            R_c2w = cam_to_world[:3,:3]
            t_c2w = cam_to_world[:3,3]

            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w.reshape(-1,1)

            camera = OpenCVCamera(height=height, width=width, K=Kmat, R=R_w2c, t=t_w2c)
            image = self.load_image(images_dir, imname)
            self.views.append({"camera" : camera, "image" : image})
    
    def get_view_center(self, view):
        return -view["camera"].R.T @ view["camera"].t 

class DataParserFactory():

    @staticmethod
    def get_parser(parser_type: str, input_path, kwargs = None) -> DataParser:
        if parser_type == "colmap":
            return ColmapDataParser(base_path=input_path, new_extension=kwargs["new_extension"])
        elif parser_type == "emap":
            return EMAPDataParser(meta_file_path=input_path)
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")