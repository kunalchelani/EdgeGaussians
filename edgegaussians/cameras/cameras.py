import torch
from typing import Union
import numpy as np
from abc import ABC, abstractmethod
from edgegaussians.utils.colmap_read_write_model import qvec2rotmat

class BaseCamera(ABC):
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_K(self):
        pass
    
    @abstractmethod
    def get_viewmat(self):
        pass
    
    @abstractmethod
    def to(self, device):
        pass
    
    def scale_translation(self, scaling_factor):
        assert hasattr(self, 't'), "Translation vector not found"
        self.t = self.t * scaling_factor
        self.viewmat = torch.cat((torch.cat((self.R, self.t.reshape(-1, 1)), dim=1), torch.tensor([[0, 0, 0, 1]])), dim = 0)
    
    def rescale_output_resolution(
        self,
        scaling_factor: Union[float, int],
        scale_rounding_mode: str = "floor",
    ) -> None:
        """Rescale the output resolution of the cameras.

        Args:
            scaling_factor: Scaling factor to apply to the output resolution.
            scale_rounding_mode: round down or round up when calculating the scaled image height and width
        """
        if isinstance(scaling_factor, (float, int)):
            scaling_factor = torch.tensor([scaling_factor]).to(self.device)
        else:
            raise ValueError(
                f"Scaling factor must be a float or int."
            )

        self.fx = self.fx * scaling_factor
        self.fy = self.fy * scaling_factor
        self.cx = self.cx * scaling_factor
        self.cy = self.cy * scaling_factor
        if scale_rounding_mode == "floor":
            self.height = (self.height * scaling_factor).to(torch.int64)
            self.width = (self.width * scaling_factor).to(torch.int64)
        elif scale_rounding_mode == "round":
            self.height = torch.floor(0.5 + (self.height * scaling_factor)).to(torch.int64)
            self.width = torch.floor(0.5 + (self.width * scaling_factor)).to(torch.int64)
        elif scale_rounding_mode == "ceil":
            self.height = torch.ceil(self.height * scaling_factor).to(torch.int64)
            self.width = torch.ceil(self.width * scaling_factor).to(torch.int64)
        else:
            raise ValueError("Scale rounding mode must be 'floor', 'round' or 'ceil'.")
        

class Camera(BaseCamera):

    def __init__(self, height, width, fx, fy, cx, cy, quat, trans, device = 'cpu', scaling_factor : float = 1.0):
            
        self.height = int(np.ceil(height * scaling_factor))
        self.width = int(np.ceil(width * scaling_factor))
        self.fx = fx * scaling_factor
        self.fy = fy * scaling_factor
        self.cx = cx * scaling_factor
        self.cy = cy * scaling_factor
        
        self.quat = quat
        self.device = device
        
        self.rot = torch.from_numpy(quat).float().reshape(1, 1, 4) 
        
        if isinstance(trans, np.ndarray):
            trans = torch.from_numpy(trans).float()
        self.t = trans # should be a 3 vec
        
        self.K =  torch.tensor([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]).float()
        rotmat = qvec2rotmat(quat)
        self.R  = torch.from_numpy(rotmat).float()
        self.viewmat = torch.cat((torch.cat((self.R, self.t.reshape(-1, 1)), dim=1), torch.tensor([[0, 0, 0, 1]])), dim = 0)
    
    def get_device(self):
        return self.device
    
    def get_K(self):
        return self.K.reshape(1, 3, 3)
    
    def get_viewmat(self):
        return self.viewmat.reshape(1, 4, 4)
    
    def to(self, device):
        self.device = device
        self.K = self.K.to(device)
        self.viewmat = self.viewmat.to(device)

class OpenCVCamera(BaseCamera):

    def __init__(self, height, width, K, R, t):
        self.height = height
        self.width = width
        self.device = 'cpu'

        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float()
        self.K = K[:3,:3]
        assert isinstance(self.K, torch.Tensor), f"K should be a torch tensor, {type(K)}"
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float()
        self.R = R
        assert isinstance(self.R, torch.Tensor), f"R should be a torch tensor {type(R)}"

        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        self.t = t
        assert isinstance(self.R, torch.Tensor), f"t should be a torch tensor {type(t)}"

        self.viewmat = torch.cat((torch.cat((self.R, self.t.reshape(-1, 1)), dim=1), torch.tensor([[0, 0, 0, 1]])), dim = 0)

    def get_K(self):
        return self.K.reshape(1, 3, 3)
    
    def get_viewmat(self):
        return self.viewmat.reshape(1, 4, 4)
    
    def to(self, device):
        self.device = device
        self.K = self.K.to(device)
        self.viewmat = self.viewmat.to(device)

