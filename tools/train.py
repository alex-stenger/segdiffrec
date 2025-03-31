#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/20 22:33
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import sys
import argparse
import copy
import logging
#import coloredlogs
import numpy as np
import torch

from torch import nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(sys.path[0]))
from model.networks.unet import YNet
from model.modules.ema import EMA
from utils.initializer import device_initializer, seed_initializer, network_initializer, optimizer_initializer, \
    sample_initializer, lr_initializer, amp_initializer
from utils.utils import plot_images, save_images, get_dataset, setup_logging, save_train_logging, save_images_with_name
from utils.checkpoint import load_ckpt, save_ckpt
from tools.test import test

logger = logging.getLogger(__name__)
#coloredlogs.install(level="INFO")


def train(rank=None, args=None):
    """
    Training
    :param rank: Device id
    :param args: Input parameters
    :return: None
    """
    lambda_rec = 1e-3
    logger.info(msg=f"[{rank}]: Input params: {args}")
    # Initialize the seed
    seed_initializer(seed_id=args.seed)
    # Sample type
    sample = args.sample
    # Network
    network = args.network
    # Run name
    run_name = args.run_name
    # Input image size
    image_size = args.image_size
    # Select optimizer
    optim = args.optim
    #lambda to regularize target recons loss
    lambda_st = args.lambda_st
    # Select activation function
    act = args.act
    # Learning rate
    init_lr = args.lr
    # Learning rate function
    lr_func = args.lr_func
    # Number of classes
    num_classes = args.num_classes
    # classifier-free guidance interpolation weight, users can better generate model effect
    cfg_scale = args.cfg_scale
    # Whether to enable conditional training
    conditional = args.conditional
    # Initialize and save the model identification bit
    # Check here whether it is single-GPU training or multi-GPU training
    save_models = True
    # Whether to enable distributed training
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # Set address and port
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # The total number of processes is equal to the number of graphics cards
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # Set device ID
        device = device_initializer(device_id=rank, is_train=True)
        # There may be random errors, using this function can reduce random errors in cudnn
        # torch.backends.cudnn.deterministic = True
        # Synchronization during distributed training
        dist.barrier()
        # If the distributed training is not the main GPU, the save model flag is False
        if dist.get_rank() != args.main_gpu:
            save_models = False
        logger.info(msg=f"[{device}]: Successfully Use distributed training.")
    else:
        distributed = False
        # Run device initializer
        device = device_initializer(device_id=args.use_gpu, is_train=True)
        logger.info(msg=f"[{device}]: Successfully Use normal training.")
    # Whether to enable automatic mixed precision training
    amp = args.amp
    # Save model interval
    save_model_interval = args.save_model_interval
    # Save model interval in the start epoch
    start_model_interval = args.start_model_interval
    # Enable data visualization
    vis = args.vis
    # Number of visualization images generated
    num_vis = args.num_vis
    # Saving path
    result_path = args.result_path
    # Create data logging path
    results_logging = setup_logging(save_path=result_path, run_name=run_name)
    results_dir = results_logging[1]
    results_vis_dir = results_logging[2]
    results_tb_dir = results_logging[3]
    
    val_path = os.path.join(results_dir,"tmp")
    if not os.path.exists(val_path):
        os.makedirs(val_path)
        print(f"Directory '{val_path}' created successfully!")

    # Dataloader
    dataset_source_path = args.dataset_source_path
    dataset_target_path = args.dataset_target_path
    dataloader_img_src, dataloader_lbl_src , dataloader_sam_src = get_dataset(args=args, dataset_path=dataset_source_path, is_train=True, distributed=distributed)
    dataloader_img_trg, _ , dataloader_sam_trg = get_dataset(args=args, dataset_path=dataset_target_path, is_train=True, distributed=distributed)

    dataloader_img_val, dataloader_lbl_val , dataloader_sam_val = get_dataset(args=args, dataset_path=args.dataset_target_path, is_train=False, is_val=True, distributed=False)

    target_name = dataset_target_path.split("/")[1]

    print(dataset_source_path)
    print(dataset_target_path)
    # Resume training
    resume = args.resume
    # Pretrain
    pretrain = args.pretrain
    # Network
    Network = network_initializer(network=network, device=device)

    # Initialize the denoising UNet
    if not conditional:
        model = Network(device=device, image_size=image_size, act=act).to(device)
    else:
        model = Network(num_classes=num_classes, device=device, image_size=image_size, act=act).to(device)
    # Distributed training
    if distributed:
        model = nn.parallel.DistributedDataParallel(module=model, device_ids=[device], find_unused_parameters=True)
    ynet = YNet()
    ynet = ynet.to(device)
    # Model optimizer
    optimizer_diff = optimizer_initializer(model=model, optim='adam', init_lr=init_lr, device=device)
    optimizer_rec = optimizer_initializer(model=ynet, optim='adam', init_lr=init_lr, device=device)
    
    # Resume training
    if resume:
        ckpt_path = None
        start_epoch = args.start_epoch
        # Determine which checkpoint to load
        # 'start_epoch' is correct
        if start_epoch is not None:
            ckpt_path = os.path.join(results_dir, f"ckpt_{str(start_epoch - 1).zfill(3)}.pt")
        # Parameter 'ckpt_path' is None in the train mode
        if ckpt_path is None:
            ckpt_path = os.path.join(results_dir, "ckpt_last.pt")
        start_epoch = load_ckpt(ckpt_path=ckpt_path, model=model, device=device, optimizer=optimizer_diff,
                                is_distributed=distributed)
        logger.info(msg=f"[{device}]: Successfully load resume model checkpoint.")
    else:
        # Pretrain mode
        if pretrain:
            pretrain_path = args.pretrain_path
            #load_ckpt(ckpt_path=pretrain_path, model=model, device=device, is_pretrain=pretrain,
            #          is_distributed=distributed)
            load_ckpt(ckpt_path=pretrain_path, model=model, device=device, is_train=False,
            is_distributed=distributed)
            logger.info(msg=f"[{device}]: Successfully load pretrain model checkpoint.")
        start_epoch = 0
    # Set harf-precision
    scaler = amp_initializer(amp=amp, device=device)
    # Loss function
    mse = nn.MSELoss()
    # Initialize the diffusion model
    diffusion = sample_initializer(sample=sample, image_size=image_size, device=device)
    # Tensorboard
    tb_logger = SummaryWriter(log_dir=results_tb_dir)
    # Train log
    save_train_logging(args, results_dir)
    # Number of dataset batches in the dataloader
    len_dataloader = len(dataloader_img_src)
    # Exponential Moving Average (EMA) may not be as dominant for single class as for multi class
    ema = EMA(beta=0.995)
    # EMA model
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    logger.info(msg=f"[{device}]: Start training.")
    # Start iterating
    src_loss_diff_list = []
    trg_loss_diff_list = []
    best_iou = 0
    for epoch in range(start_epoch, args.epochs):
        logger.info(msg=f"[{device}]: Start epoch {epoch}:")
        # Set learning rate
        current_lr = lr_initializer(lr_func=lr_func, optimizer=optimizer_diff, epoch=epoch, epochs=args.epochs,
                                    init_lr=init_lr, device=device)
        tb_logger.add_scalar(tag=f"[{device}]: Current LR", scalar_value=current_lr, global_step=epoch)
        print("current lr :", current_lr)

        # Maybe should add another lr for optim rec


        # Initialize images and labels
        pbar_img = tqdm(dataloader_img_src)
        # Initialize images and labels
        #for i, (images, labels) in enumerate(pbar):
        for i, (list_img_src, list_lbl_src, list_img_trg, list_sam_src, list_sam_trg) in enumerate(zip(pbar_img, dataloader_lbl_src, dataloader_img_trg, dataloader_sam_src, dataloader_sam_trg)):
            # The images are all resized in dataloader
            images_src = list_img_src[0].to(device)
            labels_src = list_lbl_src[0].to(device)
            images_trg = list_img_trg[0].to(device)
            sam_src = list_sam_src[0].to(device)
            sam_trg = list_sam_trg[0].to(device)
            assert list_img_src[2] == list_lbl_src[2], "The two lists are not the same" #To be sure labels are associated to img

            #print("images", images_src.shape)
            #print("labels", labels_src.shape)
            #print("sam", sam_src.shape)

            #plt.imshow(images_src[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            #plt.show()
            #plt.imshow(labels_src[0,0,:,:].detach().cpu().numpy(), cmap="gray")
            #plt.show()

            # Generates a tensor of size images.shape[0] * images.shape[0] randomly sampled time steps
            time = diffusion.sample_time_steps(labels_src.shape[0]).to(device)
            # Add noise, return as x value at time t and standard normal distribution
            x_time_source, noise_soure = diffusion.noise_images(x=labels_src, time=time)
            if(sam_trg.shape[0] != time.shape[0]) :
                tmp = torch.zeros((time.shape[0], sam_trg.shape[1], sam_trg.shape[2], sam_trg.shape[3]))
                for i in range(time.shape[0] - sam_trg.shape[0]) :
                    tmp[sam_trg.shape[0]+i] = sam_trg[i%sam_trg.shape[0]]
                sam_trg = tmp
                sam_trg = sam_trg.to(device)
            x_time_target, noise_target = diffusion.noise_images(x=sam_trg, time=time)
            # Enable Automatic mixed precision training
            # Automatic mixed precision training
            with autocast(enabled=amp):


                if epoch < 50 :
                    concat_source = torch.cat([x_time_source, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
                    predicted_noise = model.forward(concat_source, time)
                    loss_diff = mse(noise_soure, predicted_noise)

                    loss_rec = torch.zeros_like(loss_diff)

                    optimizer_diff.zero_grad()
                    # Update loss and optimizer
                    scaler.scale(loss_diff).backward() #I think should be retain_graph=True ?
                    scaler.step(optimizer_diff)
                    scaler.update()
                
                else :
                    #Learning the global distribution of mask + segmentation
                    concat_source = torch.cat([x_time_source, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
                    predicted_noise = model.forward(concat_source, time)
                    loss_diff = mse(noise_soure, predicted_noise)

                    #Learning to reconstruct target image
                    concat_target = torch.cat([x_time_target, images_trg], dim=1)
                    latent_trg = model.forward_latent(concat_target,time)
                    recons_output = ynet(latent_trg[0],latent_trg[1],latent_trg[2],latent_trg[3],latent_trg[4])

                    latent_src = model.forward_latent(concat_source,time)
                    recons_src = ynet(latent_src[0],latent_src[1],latent_src[2],latent_src[3],latent_src[4])

                    #loss_rec = lambda_rec * mse(recons_output,images_trg)
                    loss_rec = (lambda_rec) * (mse(recons_output,images_trg) + mse(recons_src,images_src))
                    #loss_rec = mse(recons_output,images_trg) + mse(recons_src,images_src)

                    optimizer_diff.zero_grad()
                    optimizer_rec.zero_grad()
                    # Update loss and optimizer
                    scaler.scale(loss_diff).backward() #I think should be retain_graph=True ?
                    scaler.scale(loss_rec).backward()
                    scaler.step(optimizer_diff)
                    scaler.step(optimizer_rec)
                    scaler.update()
                
            # EMA
            #ema.step_ema(ema_model=ema_model, model=model)

            # TensorBoard logging
            pbar_img.set_postfix(MSE=loss_diff.item())
            tb_logger.add_scalar(tag=f"[{device}]: MSE", scalar_value=loss_diff.item(),
                                 global_step=epoch * len_dataloader + i)
            src_loss_diff_list.append(loss_diff.detach().cpu().numpy().item())
            trg_loss_diff_list.append(loss_rec.detach().cpu().numpy().item())
        # Loss per epoch
        tb_logger.add_scalar(tag=f"[{device}]: Loss", scalar_value=sum(src_loss_diff_list) / len(src_loss_diff_list), global_step=epoch)

        # Saving and validating models in the main process
        if save_models:
            # Saving model, set the checkpoint name
            save_name = f"ckpt_{str(epoch).zfill(3)}"
            # Init ckpt params
            ckpt_model, ckpt_ema_model, ckpt_optimizer = None, None, None
            ckpt_model = model.state_dict()
            ckpt_optimizer = optimizer_diff.state_dict()
            # Enable visualization
            if vis:
                # images.shape[0] is the number of images in the current batch
                dataloader_img_src_test, dataloader_lbl_src_test, dataloader_sam_src_test = get_dataset(args=args, dataset_path=dataset_source_path, is_train=False, distributed=distributed)
                dataloader_img_trg_test, dataloader_lbl_trg_test, dataloader_sam_trg_test = get_dataset(args=args, dataset_path=dataset_target_path, is_train=False, distributed=distributed)
                batch_src = next(iter(dataloader_img_src_test))
                batch_src_lbl = next(iter(dataloader_lbl_src_test))
                batch_src_sam = next(iter(dataloader_sam_src_test))
                batch_trg = next(iter(dataloader_img_trg_test))
                batch_trg_lbl = next(iter(dataloader_lbl_trg_test))
                batch_trg_sam = next(iter(dataloader_sam_trg_test))
                names_source = batch_src[2]
                names_target = batch_trg[2]
                sampled_images_src = diffusion.sample_seg(model=model, img=batch_src[0].to(device), sam=batch_src_sam[0].to(device))
                sampled_images_trg = diffusion.sample_seg(model=model, img=batch_trg[0].to(device), sam=batch_trg_sam[0].to(device))
                save_images(images=sampled_images_src, path=os.path.join(results_vis_dir, f"{save_name}_src.jpg"))
                save_images(images=(batch_src[0] * 255).type(torch.uint8), path=os.path.join(results_vis_dir, f"{save_name}_img_src.jpg"))
                save_images(images=(batch_src_lbl[0] * 255).type(torch.uint8), path=os.path.join(results_vis_dir, f"{save_name}_gt_src.jpg"))
                save_images(images=sampled_images_trg, path=os.path.join(results_vis_dir, f"{save_name}_trg.jpg"))
                save_images(images=(batch_trg_lbl[0] * 255).type(torch.uint8), path=os.path.join(results_vis_dir, f"{save_name}_gt_trg.jpg"))
                #save_images_with_name(sampled_images,names_source,path=results_vis_dir)

                iteration = range(1, len(src_loss_diff_list) + 1)
                # Plotting the loss
                plt.plot(iteration, src_loss_diff_list, label='diff Loss')
                plt.plot(iteration, trg_loss_diff_list, label='rec Loss')
                plt.title('Training Loss over Iterations')
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.ylim((0,1))
                plt.legend()
                # Save the plot to an image file (e.g., PNG, PDF, etc.)
                plt.savefig(os.path.join(results_vis_dir, f"loss.png"))
                plt.close()
            
            # Save checkpoint ?
            thresh = 150 if pretrain else 0
            if epoch%50==0 or (epoch+thresh)>200 :
                num_sample = 10 
                for i, (list_img, _ ,list_sam) in enumerate(zip(dataloader_img_val, dataloader_lbl_val, dataloader_sam_val)):
                    # The images are all resized in dataloader
                    images = list_img[0].to(device)
                    sam = list_sam[0].to(device)
                    names = list_img[2]
                    for i in range(num_sample):
                        sampled_images = diffusion.sample_seg(model=model, img=images, sam=sam)
                        save_images_with_name(sampled_images,names,val_path,mutli_sample=True,sample_num=i)


                current_iou = test(pred_path=val_path, gt_path=os.path.join(dataset_target_path,"val/lbl/labels"), threshold_factor=0.5, retour=True)
                if current_iou > best_iou :
                    best_iou = current_iou
                    save_name = "best_iou_"+str(best_iou)
                    save_ckpt(epoch=epoch, save_name=save_name, ckpt_model=ckpt_model, ckpt_ema_model=ckpt_ema_model,
                            ckpt_optimizer=ckpt_optimizer, results_dir=results_dir, save_model_interval=save_model_interval,
                            start_model_interval=start_model_interval, conditional=conditional, image_size=image_size,
                            sample=sample, network=network, act=act, num_classes=num_classes)
        logger.info(msg=f"[{device}]: Finish epoch {epoch}:")

        # Synchronization during distributed training
        if distributed:
            logger.info(msg=f"[{device}]: Synchronization during distributed training.")
            dist.barrier()

    logger.info(msg=f"[{device}]: Finish training.")

    # Clean up the distributed environment
    if distributed:
        dist.destroy_process_group()


def main(args):
    """
    Main function
    :param args: Input parameters
    :return: None
    """
    if args.distributed:
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    else:
        train(args=args)


if __name__ == "__main__":
    # Training model parameters
    # required: Must be set
    # needed: Set as needed
    # recommend: Recommend to set
    parser = argparse.ArgumentParser()
    # =================================Base settings=================================
    # Set the seed for initialization (required)
    parser.add_argument("--seed", type=int, default=0)
    # Enable conditional training (required)
    # If enabled, you can modify the custom configuration.
    # For more details, please refer to the boundary line at the bottom.
    # [Note] We recommend enabling it to 'True'.
    parser.add_argument("--conditional", type=bool, default=False)
    # Set the sample type (required)
    # If not set, the default is for 'ddpm'. You can set it to either 'ddpm' or 'ddim'.
    # Option: ddpm/ddim
    parser.add_argument("--sample", type=str, default="ddim")
    # Set network
    # Option: unet_attention/cspdarkunet/unet_light
    parser.add_argument("--network", type=str, default="unet_attention")
    # File name for initializing the model (required)
    parser.add_argument("--run_name", type=str, default="df")
    # Total epoch for training (required)
    parser.add_argument("--epochs", type=int, default=3)
    # Batch size for training (required)
    parser.add_argument("--batch_size", type=int, default=2)
    # Number of sub-processes used for data loading (needed)
    # It may consume a significant amount of CPU and memory, but it can speed up the training process.
    parser.add_argument("--num_workers", type=int, default=0)
    # Input image size (required)
    parser.add_argument("--image_size", type=int, default=64)
    #Lambda to control the self_training loss for the target image
    parser.add_argument("--lambda_st", type=float, default=1e-3)
    # All images are placed in a single folder, and the path represents the image folder.
    parser.add_argument("--dataset_source_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    parser.add_argument("--dataset_target_path", type=str, default="/your/path/Defect-Diffusion-Model/datasets/dir")
    # Enable automatic mixed precision training (needed)
    # Effectively reducing GPU memory usage may lead to lower training accuracy and results.
    parser.add_argument("--amp", type=bool, default=False)
    # Set optimizer (needed)
    # Option: adam/adamw
    parser.add_argument("--optim", type=str, default="adamw")
    # Set activation function (needed)
    # Option: gelu/silu/relu/relu6/lrelu
    parser.add_argument("--act", type=str, default="gelu")
    # Learning rate (needed)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Learning rate function (needed)
    # Option: linear/cosine/warmup_cosine
    parser.add_argument("--lr_func", type=str, default="linear")
    # Saving path (required)
    parser.add_argument("--result_path", type=str, default="/your/path/Defect-Diffusion-Model/results")
    # Whether to save weight each training (recommend)
    parser.add_argument("--save_model_interval", type=bool, default=True)
    # Start epoch for saving models (needed)
    # This option saves disk space. If not set, the default is '-1'. If set,
    # it starts saving models from the specified epoch. It needs to be used with '--save_model_interval'
    parser.add_argument("--start_model_interval", type=int, default=-1)
    # Enable visualization of dataset information for model selection based on visualization (recommend)
    parser.add_argument("--vis", type=bool, default=True)
    # Number of visualization images generated (recommend)
    # If not filled, the default is the number of image classes (unconditional) or images.shape[0] (conditional)
    parser.add_argument("--num_vis", type=int, default=10)
    # Resume interrupted training (needed)
    # 1. Set to 'True' to resume interrupted training and check if the parameter 'run_name' is correct.
    # 2. Set the resume interrupted epoch number. (If not, we would select the last)
    # Note: If the epoch number of interruption is outside the condition of '--start_model_interval',
    # it will not take effect. For example, if the start saving model time is 100 and the interruption number is 50,
    # we cannot set any loading epoch points because we did not save the model.
    # We save the 'ckpt_last.pt' file every training, so we need to use the last saved model for interrupted training
    # If you do not know what epoch the checkpoint is, rename this checkpoint is 'ckpt_last'.pt
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--start_epoch", type=int, default=None)
    # Enable use pretrain model (needed)
    parser.add_argument("--pretrain", type=bool, default=False)
    # Pretrain model load path (needed)
    parser.add_argument("--pretrain_path", type=str, default="")
    # Set the use GPU in normal training (required)
    parser.add_argument("--use_gpu", type=int, default=0)

    # =================================Enable distributed training (if applicable)=================================
    # Enable distributed training (needed)
    parser.add_argument("--distributed", type=bool, default=False)
    # Set the main GPU (required)
    # Default GPU is '0'
    parser.add_argument("--main_gpu", type=int, default=0)
    # Number of distributed nodes (needed)
    # The value of world size will correspond to the actual number of GPUs or distributed nodes being used
    parser.add_argument("--world_size", type=int, default=2)

    # =====================Enable the conditional training (if '--conditional' is set to 'True')=====================
    # Number of classes (required)
    # [Note] The classes settings are consistent with the loaded datasets settings.
    parser.add_argument("--num_classes", type=int, default=10)
    # classifier-free guidance interpolation weight, users can better generate model effect (recommend)
    parser.add_argument("--cfg_scale", type=int, default=3)

    args = parser.parse_args()

    main(args)

    #if epoch > 15 : #poids encoder gelés
    #    model.freeze_weights_from_inc_to_bot3()
    #    if i==0 :
    #        print("Autoencoder Training Done, training Diffusion Model")
    #    if i==0 :
    #        for name, param in model.named_parameters():
    #            print(f"Layer: {name}, Requires Grad: {param.requires_grad}")

    #    #ATTENTION, si l'optimizer a du weight decay (typiquement AdamW), les paramètres vont tout de même être uptdate (voir algo AdamW)

    #    concat_source = torch.cat([x_time, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
    #    predicted_noise = model.forward(concat_source, time)
    #    loss_diff = mse(noise, predicted_noise)

    #    optimizer_diff.zero_grad()
    #    scaler.scale(loss_diff).backward()
    #    scaler.step(optimizer_diff)
    #    scaler.update()

    #    latent_trg = model.forward_latent(concat_target,time)
    #    recons_output = ynet(latent_trg[0],latent_trg[1],latent_trg[2],latent_trg[3],latent_trg[4])
    #    latent_src = model.forward_latent(concat_source,time)
    #    recons_src = ynet(latent_src[0],latent_src[1],latent_src[2],latent_src[3],latent_src[4])
    #    loss_rec = lambda_rec * (mse(recons_output,images_trg) + mse(recons_src,images_src))

    #else : #on pretrain l'encoder a reconstruire
    #    if i==0 :
    #        print("Pretraining AutoEncoder")
    #    concat_source = torch.cat([x_time, images_src], dim=1)
    #    concat_target = torch.cat([x_time, images_trg], dim=1)
        
    #    latent_trg = model.forward_latent(concat_target,time)
    #    recons_output = ynet(latent_trg[0],latent_trg[1],latent_trg[2],latent_trg[3],latent_trg[4])
    #    latent_src = model.forward_latent(concat_source,time)
    #    recons_src = ynet(latent_src[0],latent_src[1],latent_src[2],latent_src[3],latent_src[4])
    #    loss_rec = lambda_rec * (mse(recons_output,images_trg) + mse(recons_src,images_src))

    #    optimizer_rec.zero_grad()
    #    scaler.scale(loss_rec).backward()
    #    scaler.step(optimizer_rec)
    #    scaler.update()


    #    concat_source = torch.cat([x_time, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
    #    predicted_noise = model.forward(concat_source, time)
    #    loss_diff = mse(noise, predicted_noise)




                #    if epoch < 50 :
                #    concat_source = torch.cat([x_time, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
                #    predicted_noise = model.forward(concat_source, time)
                #    loss_diff = mse(noise, predicted_noise)
#
                #    loss_rec = torch.zeros_like(loss_diff)
#
                #    optimizer_diff.zero_grad()
                #    # Update loss and optimizer
                #    scaler.scale(loss_diff).backward() #I think should be retain_graph=True ?
                #    scaler.step(optimizer_diff)
                #    scaler.update()
                #
                #else :
                #    #Learning the global distribution of mask + segmentation
                #    concat_source = torch.cat([x_time, images_src], dim=1) #Should be of size (B,6,H,W) (img+noise)
                #    predicted_noise = model.forward(concat_source, time)
                #    loss_diff = mse(noise, predicted_noise)
#
                #    #Learning to reconstruct target image
                #    concat_target = torch.cat([x_time, images_trg], dim=1)
                #    latent_trg = model.forward_latent(concat_target,time)
                #    recons_output = ynet(latent_trg[0],latent_trg[1],latent_trg[2],latent_trg[3],latent_trg[4])
#
                #    latent_src = model.forward_latent(concat_source,time)
                #    recons_src = ynet(latent_src[0],latent_src[1],latent_src[2],latent_src[3],latent_src[4])
#
                #    #loss_rec = lambda_rec * mse(recons_output,images_trg)
                #    loss_rec = (lambda_rec) * (mse(recons_output,images_trg) + mse(recons_src,images_src))
                #    #loss_rec = mse(recons_output,images_trg) + mse(recons_src,images_src)
#
                #    optimizer_diff.zero_grad()
                #    optimizer_rec.zero_grad()
                #    # Update loss and optimizer
                #    scaler.scale(loss_diff).backward() #I think should be retain_graph=True ?
                #    scaler.scale(loss_rec).backward()
                #    scaler.step(optimizer_diff)
                #    scaler.step(optimizer_rec)
                #    scaler.update()
