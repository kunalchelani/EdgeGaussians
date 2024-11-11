import numpy as np
import os
import torch
import json
import datetime
import open3d as o3d
import ipdb

from pathlib import Path
from edgegaussians.utils.colmap_read_write_model import read_points3D_text, read_points3d_binary

from scipy.spatial.distance import cdist


class CustomLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, first_stage_epochs, lr_after_first_stage, last_epoch=-1):
        self.first_stage_epochs = first_stage_epochs
        self.lr_after_first_stage = lr_after_first_stage
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.first_stage_epochs:
            return [0 for _ in self.optimizer.param_groups]
        else:
            return [self.lr_after_first_stage for _ in self.optimizer.param_groups]
        

def get_bg_edge_pixel_ratio(loss_config, step : int , max_steps : int = 200):

    if loss_config["bg_edge_pixel_ratio_annealing"] == "constant":
        return loss_config["bg_edge_pixel_ratio_start"]
    elif loss_config["bg_edge_pixel_ratio_annealing"] == "linear":
        return loss_config["bg_edge_pixel_ratio_start"] + (loss_config["bg_edge_pixel_ratio_end"] - loss_config["bg_edge_pixel_ratio_start"]) * step / max_steps
    else:
        raise ValueError(f"Unsupported bg_edge_pixel_ratio_annealing: {loss_config['bg_edge_pixel_ratio_annealing']}")


def get_lambda_projection(loss_config, step : int , max_steps : int = 200):

    if loss_config["lambda_annealing"] == "constant":
        return loss_config["lambda_start"]
    elif loss_config["lambda_annealing"] == "linear":
        return loss_config["lambda_start"] + (loss_config["lambda_end"] - loss_config["lambda_start"]) * step / max_steps
    else:
        raise ValueError(f"Unsupported lambda_annealing: {loss_config['lambda_annealing']}")
    
        
def get_optimizers_schedulers(model, config):

    means_optimizer = torch.optim.Adam([model.gauss_params["means"]], lr=config["means"]["start_lr"])
    means_scheduler = torch.optim.lr_scheduler.MultiStepLR(means_optimizer, milestones=config["means"]["milestones"], gamma=config["means"]["gamma"])
    
    scales_optimizer = torch.optim.Adam([model.gauss_params["scales"]], lr=config["scales"]["start_lr"])
    scales_scheduler = CustomLRScheduler(scales_optimizer, first_stage_epochs = config["scales"]["start_at_epoch"] , lr_after_first_stage = config["scales"]["start_lr"])
    
    quats_optimizer = torch.optim.Adam([model.gauss_params["quats"]], lr=config["quats"]["start_lr"])
    quats_scheduler = CustomLRScheduler(quats_optimizer, first_stage_epochs = config["quats"]["start_at_epoch"], lr_after_first_stage = config["quats"]["start_lr"])

    opacities_optimizer = torch.optim.Adam([model.gauss_params["opacities"]], config["opacities"]["start_lr"])
    opacities_scheduler = CustomLRScheduler(opacities_optimizer, first_stage_epochs = config["opacities"]["start_at_epoch"], lr_after_first_stage = config["opacities"]["start_lr"])

    # return the combined optimizer
    optimizers = {"means":means_optimizer, "scales":scales_optimizer, "quats":quats_optimizer, "opacities":opacities_optimizer}
    schedulers = {"scales":scales_scheduler, "means":means_scheduler, "opacities":opacities_scheduler, "quats":quats_scheduler}
    return optimizers, schedulers

    
def save_model(model, output_dir, epoch):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    weights_file_path = os.path.join(output_dir, f"epoch{epoch}.pth")
    if os.path.exists(weights_file_path):
        append = datetime.datetime.now().strftime("%Y%m%d%H%M%S")[4:]
        weights_file_path = os.path.join(output_dir, f"epoch{epoch}{append}.pth")
    torch.save(model.state_dict(), weights_file_path)


def remove_old_models(output_dir):
    if not os.path.exists(output_dir):
        return
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))