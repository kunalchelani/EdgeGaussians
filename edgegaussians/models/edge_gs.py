import torch
import numpy as np
import time
import ipdb

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field
from gsplat import rasterization
from dacite import from_dict
from sklearn.neighbors import NearestNeighbors

from edgegaussians.models.losses import MaskedL1Loss, WeightedL1Loss
from edgegaussians.cameras.cameras import BaseCamera
from edgegaussians.utils.misc_utils import unravel_index, random_quat_tensor, quats_to_rotmats_tensor
from edgegaussians.utils.io_utils import write_gaussian_params_as_ply
@dataclass
class EdgeGaussianSplattingConfig:

    if_duplicate_high_pos_grad: bool = True
    dup_threshold_type: str = "percentile"
    dup_threshold_value: float = 0.95
    dup_factor: int = 2
    dup_high_pos_grads_at_epoch: list = field(default_factory=lambda: [36, 46, 51, 76, 101, 126, 151])

    if_cull_low_opacity: bool = True
    cull_opacity_type: str = "absolute"
    cull_opacity_value: float = 0.05
    cull_opacity_at_epoch : list = field(default_factory=lambda: [80,160])

    if_cull_wayward: bool = True
    cull_wayward_method: str = "mean_distance"
    cull_wayward_num_neighbors: int = 10
    cull_wayward_threshold_type: str = "percentile_top"
    cull_wayward_threshold_value: float = 0.05
    cull_wayward_at_epoch : list = field(default_factory=lambda: [51,101,151])

    init_random_init: bool = False
    init_dup_rand_noise_scale: float = 0.05
    init_min_num_gaussians: int = 5000
    init_scales_type: str = "constant"
    init_scales_val: float = 0.005
    init_opacity_type: str = "constant"
    init_opacity_val: float = 0.08

    if_cull_gaussians_not_projecting : bool = True
    cull_gaussians_not_projecting_at_epoch : list = field(default_factory=lambda: [50,100,150])
    cull_gaussians_not_projecting_threshold : float = 0.35

    edge_detection_threshold: float = 0.5
    rasterize_mode = "antialiased"

    if_reset_opacity : bool = False
    reset_opacity_at_epoch : list = field(default_factory=lambda: [100])
    reset_opacity_value : float = 0.08


'''
Much of the following class is inspired from the Splatfacto model in nerfstudio.
'''

class EdgeGaussianSplatting(torch.nn.Module):

    def __init__(self, device = 'cuda'):
        self.device = device
        super().__init__()

    def poplutate_params(self, seed_points = None, viewcams = None, config = None):

        assert seed_points is not None, "Seed points need to be provided"
        assert viewcams is not None, "Viewcams need to be provided"
        assert config is not None, "Config needs to be provided"

        config = from_dict(data_class=EdgeGaussianSplattingConfig, data=config)
        self.config = config

        self.seed_points = seed_points

        means = torch.nn.Parameter(self.seed_points)    
        constant_scale = torch.Tensor([config.init_scales_val ]).float()
        scales = torch.nn.Parameter(torch.log(constant_scale.repeat(means.shape[0], 3)))
        
        num_points = means.shape[0]
        self.viewcams = viewcams
        
        self.bg_pixels = []
        self.edge_pixels = []
        self.edge_masks = []
        self.absgrads = torch.zeros(num_points).to(self.device)
        self.absgrads_normalize_factor = 1.0
        
        self.crop_box = None # Can be used for cropping the gaussians to a specific region

        opacities = torch.nn.Parameter(torch.logit(config.init_opacity_val * torch.ones(num_points, 1)))
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
            }
        )
        self.step = 0
    
    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]
    
    @property
    def opacities(self):
        return self.gauss_params["opacities"]
    
    def get_gaussian_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
            # Here we explicitly use the means, scales as parameters so that the user can override this function and
            # specify more if they want to add more optimizable params to gaussians.
            return {
                name: [self.gauss_params[name]]
                for name in ["means", "scales", "quats", "opacities"]
        }
    
    # required for enforcing the geometric constraints
    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
    # Functions required for the occlusion aware projection loss - is slow and can be optimized
    def compute_image_masks(self, gt_images):

        for image in gt_images:
            # print("Computing gt image shape ", image.shape)
            edge_mask = image >= self.config.edge_detection_threshold
            self.edge_masks.append(edge_mask)

        print("Computed masks for all images")

    def sample_pixels_for_loss(self, image_idx, ratio_edge_to_bg:float = 1):
        """
        Sample pixels from the image for computing the loss
        """
        bg_pixels = self.bg_pixels[image_idx]
        edge_pixels = self.edge_pixels[image_idx]

        num_bg_pixels = int(ratio_edge_to_bg * len(edge_pixels))
        bg_pixels = bg_pixels[torch.randperm(len(bg_pixels))[:num_bg_pixels]]
    

        return edge_pixels, bg_pixels
    
    # Weighted L1 loss for the occlusion aware projection loss
    def compute_weight_masks(self):
        assert hasattr(self, "edge_masks"), "Edge masks need to be computed first"
        assert self.edge_masks is not None, "Edge masks need to be computed first"

        self.weight_masks = []
        for edge_mask in self.edge_masks:
            
            num_edge_pixels = edge_mask.sum()
            num_bg_pixels = (~edge_mask).sum()
            
            edge_weight = num_bg_pixels / (num_edge_pixels + num_bg_pixels)
            bg_weight = num_edge_pixels / (num_edge_pixels + num_bg_pixels)

            weight_mask = torch.zeros_like(edge_mask, dtype = torch.float)
            weight_mask[edge_mask] = edge_weight
            weight_mask[~edge_mask] = bg_weight
            self.weight_masks.append(weight_mask)


    # Functions required for obtaining the rendered image from a single view - should be batched for faster training
    def get_outputs(self, camera: BaseCamera) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]

            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            quats_crop = quats_crop / torch.norm(quats_crop, dim=-1, keepdim=True)
        else:
            opacities_crop = self.opacities
            means_crop = self.means

            scales_crop = self.scales
            quats_crop = self.quats


        BLOCK_WIDTH = 16

        viewmat = camera.get_viewmat()
        K = camera.get_K()

        # W, H = int(camera.width.item()), int(camera.height.item())
        W, H = camera.width, camera.height
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        render_mode = "RGB"
        colors_crop = torch.ones(means_crop.shape[0], 3).cuda()

        # ipdb.set_trace()
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
        )

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()

        self.xys = info["means2d"]  # [1, N, 2]

        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        rgb = render[:, ..., :3]
        rgb = torch.clamp(rgb, 0.0, 1.0)
        depth_im = None

        return {
            "rgb": rgb.squeeze(0),  
            "depth": depth_im, 
            "accumulation": alpha.squeeze(0),
        }
    
    def compute_projection_loss(self, output_image, gt_image, image_index = None, strategy = "bg_edge_ratio", bg_edge_pixel_ratio = 1.0, loss_type: str = "l1"):

        if strategy == "whole":
            if loss_type == "l1":
                criterion = torch.nn.functional.l1_loss
            elif loss_type == "l2":
                criterion = torch.nn.functional.mse_loss
            loss = criterion(output_image, gt_image)
            return loss

        elif strategy == "bg_edge_ratio":
            masked_l1_loss = MaskedL1Loss()
            edge_loss = masked_l1_loss(output_image, gt_image, self.edge_masks[image_index])
            
            # sample pixels from bg
            num_bg_pixels = int(bg_edge_pixel_ratio * self.edge_masks[image_index].sum())
            bg_mask = ~self.edge_masks[image_index]
            bg_mask_1 = torch.where(bg_mask)[0]
            bg_flat_select_1 = torch.randperm(len(bg_mask_1))[:num_bg_pixels]
            indices = unravel_index(bg_flat_select_1, bg_mask.shape)

            bg_mask_final = torch.zeros_like(bg_mask, dtype = torch.bool)
            bg_mask_final[indices[:,0], indices[:,1]] = True
            
            # compute loss for edges and sample bg
            bg_loss = masked_l1_loss(output_image, gt_image, bg_mask_final)
            loss = edge_loss + bg_loss

        elif strategy == "weighted":
            weighted_l1_loss = WeightedL1Loss()
            weight_mask = self.weight_masks[image_index]
            loss = weighted_l1_loss(output_image, gt_image, weight_mask)
        
        else:
            raise ValueError(f"Unknown projection loss strategy: {strategy}")

        return loss

    def update_nearest_neighbors(self):
        # compute the nearest neighbors for each point
        k = self.dir_loss_num_nn
        # check for nan values in the means and replace them with 0
        points = self.means.data
        if torch.isnan(points).sum() > 0:
            points[torch.isnan(points)] = 0
            print("Points with nan values ", torch.isnan(points).sum())

        start_time = time.time()
        if self.dir_loss_enforce_method != 'enforce_half':
            _, indices = self.k_nearest_sklearn(points, k+1)
            
        elif self.dir_loss_enforce_method == 'enforce_half':
            _, indices = self.k_nearest_sklearn(points, 2*k+1)
        
        end_time = time.time()
        # print(f"Time taken to compute nearest neighbors {end_time - start_time}")
        self.nn_indices = indices[:,1:]

    def compute_direction_loss(self, visualize = False):
        
        k = self.dir_loss_num_nn
        inds = torch.from_numpy(self.nn_indices).long()
        
        # get the major dorections for each gaussian
        rotmats = quats_to_rotmats_tensor(self.quats)
        scales = torch.exp(self.scales)
        argmax_scales = torch.argmax(torch.abs(scales), dim=-1)
        rotmats = rotmats.to(self.device)
        major_dirs = rotmats[torch.arange(self.num_points), :, argmax_scales]

        # get the directions towards the nearest neighbors
        neighbor_dirs = self.means[:, None, :] - self.means[inds]
        neighbor_dirs = neighbor_dirs / torch.norm(neighbor_dirs, dim=-1, keepdim=True)

        if self.dir_loss_enforce_method != 'enforce_half':
            alignment = torch.abs(torch.sum(major_dirs[:, None, :] * neighbor_dirs, dim=-1))
            mean_alignment = torch.mean(alignment, dim=-1)

        elif self.dir_loss_enforce_method == 'enforce_half':
            alignment = torch.abs(torch.sum(major_dirs[:, None, :] * neighbor_dirs, dim=-1))
            alignment_sorted, _ = torch.sort(alignment, dim=-1, descending=True)
            mean_alignment = torch.mean(alignment_sorted[:,:k], dim=-1)

        loss = 1.0 - torch.mean(mean_alignment)

        return loss

    def compute_ratio_loss(self):
        # get the ratio second largest to the largest scale for each gaussian
        scales = torch.exp(self.scales)
        sorted_scales, _ = torch.sort(scales, dim=-1, descending=True)
        ratio = sorted_scales[:, 1] / sorted_scales[:, 0]
        return torch.mean(ratio)
    
    
    ## Culling and duplication ##
    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state


    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()


    def cull_gaussians(self, optimizers, cull_mask, reset_rest = True):
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = param[~cull_mask]
        
        if reset_rest:
            self.reset_opacities()

        self.remove_from_all_optim(optimizers, cull_mask)
        self.absgrads = self.absgrads[~cull_mask]
        
        num_culled = torch.sum(cull_mask).item()
        print(f"Culled {num_culled} gaussians")
    
    def reset_opacities(self):
        self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=self.config.reset_opacity_value
                )

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            
            dup_exp_avg_list = [torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims) for i in range(self.config.dup_factor-1)]
            dup_exp_avg_sq_list = [torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims) for i in range(self.config.dup_factor-1)]

            param_state["exp_avg"] = torch.cat(
                [param_state["exp_avg"]] + dup_exp_avg_list,
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [param_state["exp_avg_sq"]] + dup_exp_avg_sq_list,
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n = 1):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers[group], dup_mask, param, n)

    
    def dup_gaussians(self, optimizers, dup_mask):
        for name, param in self.gauss_params.items():
            if name == "means":
                # add small gaussian noise to the duplicated points
                dup_means_list = [param[dup_mask] for i in range(self.config.dup_factor-1)]
                dup_means_tensor = torch.cat(dup_means_list, dim=0)
                dup_means_tensor += torch.randn_like(dup_means_tensor) * self.config.init_dup_rand_noise_scale
                self.gauss_params[name] = torch.cat([param, dup_means_tensor],  dim=0)
            else:
                concat_list = [param] + [param[dup_mask] for i in range(self.config.dup_factor-1)]
                self.gauss_params[name] = torch.cat(concat_list, dim=0)

        self.dup_in_all_optim(optimizers, dup_mask, n=1)
        num_dup = torch.sum(dup_mask).item()
        print(f"Duplicated {num_dup} gaussians")

    
    def cull_gaussians_opacity(self, optimizers):
        
        if self.config.cull_opacity_type == "percentile":
            cull_thresh = torch.quantile(torch.sigmoid(self.opacities), self.config.cull_opacity_value)
            cull_mask = torch.sigmoid(self.opacities) < cull_thresh
        elif self.config.cull_opacity_type == "absolute":
            cull_mask = torch.sigmoid(self.opacities) < self.config.cull_opacity_value
        
        cull_mask = cull_mask.squeeze()
        # opacity_cull_mask = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        
        self.cull_gaussians(optimizers, cull_mask)


    def duplicate_all_existing_gaussians(self, optimizers):
        # duplicate the gaussians
        num_gaussians = self.means.shape[0]
        # mask should be of size (num_gaussians,)
        dup_mask = torch.ones(num_gaussians, dtype=torch.bool)
        self.dup_gaussians(optimizers, dup_mask)

    def cull_wayward(self, 
                     optimizers,
                     vis_before_culling : bool = False, 
                     vis_after_culling : bool = False):
        
        num_neighbors = self.config.cull_wayward_num_neighbors
        distances, indices = self.k_nearest_sklearn(self.means.data, num_neighbors)
        inds = torch.from_numpy(indices).long()
        
        dirs_to_neighbors = self.means[:, None, :] - self.means[inds]
        dirs_to_neighbors = dirs_to_neighbors / torch.norm(dirs_to_neighbors, dim=-1, keepdim=True)
        
        if self.config.cull_wayward_method == 'pca_ratio':
            
            U, S, V = torch.pca_lowrank(dirs_to_neighbors, q = 3)
            cns = S[:, 2] / S[:, 1] # If this is low that means the variance is low in the direction of the third principal component and these are the points we need
            _, sorted_inds = torch.sort(cns, descending=False)
            cull_percentile = self.config.cull_wayward_threshold_value
            num_points_to_remove = cull_percentile * len(sorted_inds)
            wayward_cull_mask = torch.zeros_like(cns, dtype = torch.bool)
            wayward_cull_mask[sorted_inds[:num_points_to_remove]] = True
            vis_colors = torch.stack([cns, cns, cns], dim=-1).detach().cpu().numpy()

        else:
            if self.config.cull_wayward_method == 'mean_distance':
                
                dists = np.mean(distances, axis=-1)
                dists_normalized = (dists - dists.min()) / (dists.max() - dists.min())

            elif self.config.cull_wayward_method == 'max_distance':
                dists = np.max(distances, axis=-1)
                dists_normalized = (dists - dists.min()) / (dists.max() - dists.min())

            if self.config.cull_wayward_threshold_type == "percentile_top":

                cull_percentile = 1 - self.config.cull_wayward_threshold_value
                cull_beyond_dist = torch.quantile(torch.from_numpy(dists), cull_percentile, interpolation='lower').item()
                wayward_cull_mask = torch.zeros_like(torch.from_numpy(dists), dtype = torch.bool)
                wayward_cull_mask[dists > cull_beyond_dist] = True

            elif self.config.cull_wayward_threshold_type == "absolute":
                cull_thresh = self.config.cull_wayward_threshold_value
                wayward_cull_mask = torch.from_numpy(dists) > cull_thresh
            
            vis_colors = np.hstack([dists_normalized[:, None], dists_normalized[:, None], dists_normalized[:, None]])

    def duplicate_high_pos_gradients(self, optimizers):
        
        # grads = self.xys.absgrad[0].norm(dim=-1)
        grads = self.absgrads / self.absgrads_normalize_factor
        absgrads_median = torch.median(grads)
        absgrads_mean = torch.mean(grads)
        absgrads_std = torch.std(grads)
        absgrads_80percentile = torch.quantile(grads, 0.8, interpolation='lower')
        absgrads_90percentile = torch.quantile(grads, 0.9, interpolation='lower')
        # print(f"Mean absgrads {absgrads_mean}, Median absgrads {absgrads_median}, Std absgrads {absgrads_std}")
        # print(f"80 percentile absgrads {absgrads_80percentile}, 90 percentile absgrads {absgrads_90percentile}")

        grads_n = (grads - grads.min()) / (grads.max() - grads.min())
        grads_n = grads_n.detach().cpu().numpy()

        if self.config.dup_threshold_type == "percentile_top":
            duplicate_top_percentile = self.config.dup_threshold_value
            num_quantiles = int(1 / duplicate_top_percentile)
            quantiles = torch.zeros(num_quantiles)
            for i in range(1, num_quantiles):
                quantiles[i] = torch.quantile(grads, i/num_quantiles, interpolation='lower')
            
            thresh = quantiles[-1]
            # duplicate the points with top percentile
            dup_mask = grads_n > thresh
            
        elif self.config.dup_threshold_type == "absolute":
            thresh = self.config.dup_threshold_value
            # ipdb.set_trace()
            dup_mask = torch.from_numpy(grads_n > thresh)
        
        self.dup_gaussians(optimizers, dup_mask)
        self.reset_absgrads()
    
    def cull_gaussians_not_projecting(self, optimizers, min_projecting_fraction = 0.1):
        
        num_gs = self.means.shape[0]
        num_frames = len(self.viewcams)

        gs_visib_matrix = torch.zeros(num_gs, num_frames, dtype=torch.bool)
        
        for idx, viewcam in enumerate(self.viewcams):

            P = viewcam.K.cpu() @ viewcam.viewmat.cpu()[:3, :4]
            w, h = viewcam.width, viewcam.height
            gaussian_means_h = torch.cat([self.means.detach().cpu(), torch.ones(num_gs, 1)], dim=-1)
            projected_means = torch.matmul(P, gaussian_means_h.t()).t()
            projected_means = projected_means[:, :2] / projected_means[:, 2:]
            projected_means_r = projected_means.round().long()
            good_inds = (projected_means_r[:, 0] >= 0) & (projected_means_r[:, 0] < w) & (projected_means_r[:, 1] >= 0) & (projected_means_r[:, 1] < h)
            projecting_within = projected_means_r[good_inds]
            projecting_on_edge = self.edge_masks[idx][projecting_within[:, 1], projecting_within[:, 0]]
            gs_visib_matrix[good_inds, idx] = projecting_on_edge
        
        mean_projections = torch.mean(gs_visib_matrix.float(), dim=1)
        # ipdb.set_trace()
        cull_mask = mean_projections < min_projecting_fraction
        self.cull_gaussians(optimizers, cull_mask)

    def reset_absgrads(self):
        self.absgrads = torch.zeros(self.means.shape[0]).to(self.device)
        self.absgrads_normalize_factor = 1
    
    def update_absgrads(self):
        if self.absgrads.shape[0] != self.means.shape[0]:
            self.absgrads = torch.zeros(self.means.shape[0]).to(self.device)
            self.absgrads_normalize_factor = 1

        self.absgrads += self.xys.absgrad[0].norm(dim=-1)
        self.absgrads_normalize_factor += 1

    # Forward and load state dict

    def forward(self, idx):

        camera = self.viewcams[idx]
        outputs = self.get_outputs(camera)
        self.step += 1

        return outputs
    
    def load_state_dict(self, state_dict):
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(state_dict["gauss_params.means"]),
                "scales": torch.nn.Parameter(state_dict["gauss_params.scales"]),
                "quats": torch.nn.Parameter(state_dict["gauss_params.quats"]),
                "opacities": torch.nn.Parameter(state_dict["gauss_params.opacities"]),
            }
        )
    
    def export_as_ply(self, ply_path):
        
        scales = torch.exp(self.scales).detach().cpu().numpy()
        opacities = torch.sigmoid(self.opacities).detach().cpu().numpy()
        means = self.means.detach().cpu().numpy()
        quats = self.quats.detach().cpu().numpy()

        write_gaussian_params_as_ply(means, scales, quats, opacities, ply_path)