import os
import torch
import time
import argparse
import ipdb

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from edgegaussians.data.dataset import InputDataset
from edgegaussians.utils import parse_utils, train_utils, data_utils
from edgegaussians.vis import vis_utils
from edgegaussians.models.edge_gs import EdgeGaussianSplatting
from edgegaussians.data.dataparsers import DataParserFactory

def train_epoch(model, 
                dataloader, 
                optimizers,
                device,
                summary_writer,
                epoch,
                num_epochs,
                projection_loss_config,
                orientation_loss_config,
                weights_update_freq = 1,
            ):

    lambda_projection_loss = lambda_dir_loss = lambda_ratio_loss = 1.0
    bg_edge_pixel_ratio = train_utils.get_bg_edge_pixel_ratio(loss_config=projection_loss_config, 
                                                  step = epoch, 
                                                  max_steps = num_epochs)
    lambda_projection_loss = train_utils.get_lambda_projection(loss_config=projection_loss_config,
                                                   step = epoch,
                                                   max_steps = num_epochs)

    direction_loss_start_at = orientation_loss_config["start_dir_loss_at_epoch"]
    ratio_loss_start_at = orientation_loss_config["start_ratio_loss_at_epoch"]
    ratio_loss_scale_factor = orientation_loss_config["ratio_loss_scale_factor"]
    direction_loss_scale_factor = orientation_loss_config["dir_loss_scale_factor"]
    apply_ratio_loss = False
    apply_direction_loss = False

    div = 1
    if epoch > direction_loss_start_at:
        div += direction_loss_scale_factor
    if epoch > ratio_loss_start_at:
        div += ratio_loss_scale_factor
    else:
        div = 1
    
    for _,opt in optimizers.items():
        opt.zero_grad()

    avg_loss = 0

    sampling_whole_num_epochs_ratio = projection_loss_config["sampling_whole_num_epochs_ratio"]
    pixel_sampling = projection_loss_config["loss_before_alternating"]

    if epoch > projection_loss_config["start_alternating_at_epoch"]:
        check_pixel_sampling = True
    else:
        check_pixel_sampling = False

    if epoch > direction_loss_start_at:
        apply_direction_loss = True
    
    if epoch > ratio_loss_start_at:
        apply_ratio_loss = True

    for i, data in enumerate(dataloader):

        if check_pixel_sampling:
            if model.step % sampling_whole_num_epochs_ratio == 0:
                pixel_sampling = projection_loss_config['less_freq_loss']
            else:
                pixel_sampling = projection_loss_config['more_freq_loss']
        
        # get rendered image
        idx = data['idx']
        output = model(idx)

        output_image = output['rgb']
        output_image = output_image[:,:,0].unsqueeze(0)

        # get ground truth image
        gt_image = data['image']/255.0
        gt_image = gt_image.to(device)
        
        # compute projection loss
        projection_loss = model.compute_projection_loss(output_image[0,:,:], gt_image[0,:,:], 
                                                        image_index=idx,
                                                        strategy=pixel_sampling,
                                                        bg_edge_pixel_ratio = bg_edge_pixel_ratio)
        
        summary_writer.add_scalar('Projection loss', projection_loss.item(), epoch)

        projection_loss_ = lambda_projection_loss * projection_loss
        avg_loss += projection_loss.item()

        projection_loss_.backward()
        model.update_absgrads()

        for param,opt in optimizers.items():
            opt.step()
            opt.zero_grad()

        if apply_direction_loss:
            if model.step % 5 == 0:
                model.update_nearest_neighbors()
                direction_loss = model.compute_direction_loss()
                summary_writer.add_scalar('Direction loss', direction_loss.item(), epoch)
                lambda_dir_loss = (avg_loss * direction_loss_scale_factor) / direction_loss.item()
                direction_loss_ = lambda_dir_loss * direction_loss
                direction_loss_.backward()
                for param,opt in optimizers.items():
                    if param in ["means", "scales", "quats"]:
                        opt.step()
                        opt.zero_grad()
        
        if apply_ratio_loss:
            if model.step % 5 == 0:
                ratio_loss = model.compute_ratio_loss()
                summary_writer.add_scalar('Ratio loss', ratio_loss.item(), epoch)
                lambda_ratio_loss = (avg_loss * ratio_loss_scale_factor) / ratio_loss.item()
                ratio_loss_ =  lambda_ratio_loss * ratio_loss
                ratio_loss_.backward()
                for param,opt in optimizers.items():
                    if param in ["means", "scales", "quats"]:
                        opt.step()
                        opt.zero_grad()
                

    avg_loss /= len(dataloader)

    if epoch % 5 == 0:
        # Write an image grid to tensorboard
        summary_writer.add_image('Output Image', output_image, epoch)
        summary_writer.add_image('GT Image', gt_image, epoch)
        
    return avg_loss


def train(model, config, dataloader, log_dir, output_dir, device):
    summary_writer = SummaryWriter(log_dir)
    
    optim_config = config["optim"]
    loss_config = config["loss"]
    num_epochs = config["num_epochs"]
    weights_update_freq = config["weights_update_freq"]

    optimizers, schedulers = train_utils.get_optimizers_schedulers(model = model,
                                                       config = optim_config)
    print("Optimizers and schedulers created")

    projection_loss_config = loss_config["projection_losses"]
    orientation_loss_config = loss_config["orientation_losses"]

    model.dir_loss_num_nn = orientation_loss_config["dir_loss_num_nn"]
    model.dir_loss_enforce_method = orientation_loss_config["dir_loss_enforce_method"]
    model.update_nearest_neighbors()
    
    with tqdm(total=num_epochs, desc=f"Training", unit='epoch') as pbar:
        for epoch in range(num_epochs):
            
            avg_loss = train_epoch(model, 
                                dataloader, 
                                optimizers, 
                                device,
                                summary_writer,
                                epoch, 
                                num_epochs,
                                projection_loss_config,
                                orientation_loss_config,
                                weights_update_freq)
            
            pbar.set_postfix({'Loss': avg_loss, 
                              "Num Gaussians": model.gauss_params["means"].shape[0]})
            pbar.update(1)
            update_nn = False
            reset_absgrads = False

            for _,sch in schedulers.items():
                sch.step()

            if  (model.config.if_duplicate_high_pos_grad) and (epoch in model.config.dup_high_pos_grads_at_epoch):
                model.duplicate_high_pos_gradients(optimizers)
                update_nn = True
                reset_absgrads = True
                summary_writer.add_scalar('num_gaussians', model.gauss_params["means"].shape[0], epoch)

            if model.config.if_cull_gaussians_not_projecting and (epoch in model.config.cull_gaussians_not_projecting_at_epoch):
                model.cull_gaussians_not_projecting(optimizers, model.config.cull_gaussians_not_projecting_threshold)
                update_nn = True
                reset_absgrads = True
                summary_writer.add_scalar('num_gaussians', model.gauss_params["means"].shape[0], epoch)
                
            if (model.config.if_cull_low_opacity) and (epoch in model.config.cull_opacity_at_epoch):
                model.cull_gaussians_opacity(optimizers)
                update_nn = True
                reset_absgrads = True
                summary_writer.add_scalar('num_gaussians', model.gauss_params["means"].shape[0], epoch)

            if  (model.config.if_cull_wayward) and (epoch in model.config.cull_wayward_at_epoch):
                model.cull_wayward(optimizers)
                update_nn = True
                reset_absgrads = True
                summary_writer.add_scalar('num_gaussians', model.gauss_params["means"].shape[0], epoch)

            if model.config.if_reset_opacity and (epoch in model.config.reset_opacity_at_epoch):
                model.reset_opacities(optimizers)
                update_nn = True
                reset_absgrads = True
            
            if update_nn:
                model.update_nearest_neighbors()

            if reset_absgrads:
                model.reset_absgrads()
            
    
    train_utils.save_model(model, output_dir, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the config file")
    parser.add_argument("--ckpt_path", type=str, help="Load from pretrained checkpoint at this path", default=None)
    parser.add_argument("--scene_name", type=str, help="Name of the experiment", default=None)
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun the training", default=False)
    
    args = parser.parse_args()

    # Data Parse
    config_file = args.config_file
    scene_name = args.scene_name
    force_rerun = True if args.force_rerun else False

    # Load config
    model_config, training_config, data_config, output_config = parse_utils.get_configs(config_file)
    
    # get data parser
    dataparser, images_dir, seed_points_path = parse_utils.parse_data(data_config, scene_name)

    # init seed points
    if not model_config["init_random_init"]:
        seed_points = data_utils.init_seed_points_from_file(model_config, seed_points_path)
    else:
        num_seed_points = model_config["init_min_num_gaussians"]
        if "random_init_box_center" in model_config:
            box_center = model_config["random_init_box_center"]
        else:
            box_center = 0.5

        box_size = model_config["random_init_box_size"]
        num_seed_points = model_config["init_min_num_gaussians"]
        seed_points = data_utils.init_seed_points_random(num_seed_points, box_center, box_size)

    # initialize views and get scale factor if needed
    parser_type = data_config["parser_type"]
    image_res_scaling_factor = data_config["image_res_scaling_factor"]

    data_utils.init_views(dataparser, 
                             images_dir, 
                             parser_type = parser_type,
                             image_res_scaling_factor = image_res_scaling_factor)

    # Scale and translate seed points
    if (data_config["scale_scene_unit"]):
        # get scale from cameras
        rotmats = [view['camera'].R for view in dataparser.views]
        tvecs = [view['camera'].t for view in dataparser.views]
        scale = data_utils.get_scale_from_cameras(rotmats, tvecs)
        # if seed points exist, get scale from seed points
        
        if seed_points is not None:
            points_scale = data_utils.get_scale_from_points(seed_points, min_percentile=0.05, max_percentile=0.95)
            # use the maxiumum to scale both seed points and views
            scale = max(scale, points_scale)

        # scale points and cameras
        seed_points = seed_points * 1/scale
        for view in dataparser.views:
            view['camera'].scale_translation(1/scale)
    
    # ipdb.set_trace()
    # vis_utils.vis_views_as_open3d_lineset(dataparser.views)
    
    # Set up the model and the dataloader and appropriate paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views = [view['camera'] for view in dataparser.views]
    gt_images = [view['image']/255.0 for view in dataparser.views]

    for view in views:
        view.to(device)

    model = EdgeGaussianSplatting()
    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path))
    else:
        model.poplutate_params(seed_points=seed_points, viewcams=views, config=model_config)
    
    print("Model populated")
    
    model.compute_image_masks(gt_images)
    model.compute_weight_masks()
    model.to(device)

    # Create the dataloader
    dataset = InputDataset(dataparser=dataparser)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Loaded {len(dataloader)} samples in the dataloader")

    output_base = output_config["output_dir"]

    edge_detector = data_config["edge_detection_method"]
    exp_name = output_config["exp_name"] + "_" + edge_detector
    output_dir = os.path.join(output_base, exp_name, scene_name)

    log_dir = os.path.join(output_config["log_dir"], exp_name, scene_name,)
    start_time = time.time()

    # Train the model
    num_epochs = training_config["num_epochs"]
    max_epochs_weights_file = os.path.join(output_dir, f"{exp_name}_epoch{num_epochs-1}.pth")
    if os.path.exists(max_epochs_weights_file):
        if not force_rerun:
            print(f"Model already trained for {num_epochs} epochs. Exiting")
            return 0
    
    train(model=model,
          config = training_config,
          dataloader=dataloader, 
          log_dir=log_dir,
          output_dir=output_dir,
          device=device)

    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds")
    with open(os.path.join(output_dir, "time.txt"), "w") as f:
        f.write(f"Training took {end_time - start_time} seconds")

    if output_config["export_ply"]:
        output_ply_path = os.path.join(output_dir, "gaussians_all.ply")
        model.export_as_ply(output_ply_path)

if __name__ == "__main__":
    main()