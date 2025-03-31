#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/15 17:12
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import os
import logging
import shutil
import time

#import coloredlogs
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)
#coloredlogs.install(level="INFO")


def plot_images(images, fig_size=(64, 64)):
    """
    Draw images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    plt.imshow(X=torch.cat([torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def plot_one_image_in_images(images, fig_size=(64, 64)):
    """
    Draw one image in images
    :param images: Image
    :param fig_size: Draw image size
    :return: None
    """
    plt.figure(figsize=fig_size)
    for i in images.cpu():
        plt.imshow(X=i)
        plt.imshow()


def save_images_with_name(images, names, path, mutli_sample=False,sample_num=0,**kwargs):
    """
    Save images
    :param images: Image
    :param path: Save path
    :param kwargs: Other parameters
    :return: None
    """
    for i in range(images.shape[0]) :
        if not mutli_sample :
            img = images[i].to("cpu").numpy()
            name = names[i]
            plt.imsave(os.path.join(path,name),img[0,:,:], cmap="gray")
        else :
            img = images[i].to("cpu").numpy()
            name = names[i].split(".")[0]+"_"+str(sample_num)+".png"
            plt.imsave(os.path.join(path,name),img[0,:,:], cmap="gray")

def save_images(images, path, **kwargs):
    """
    Save images
    :param images: Image
    :param path: Save path
    :param kwargs: Other parameters
    :return: None
    """
    grid = torchvision.utils.make_grid(tensor=images, **kwargs)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(obj=image_array)
    im.save(fp=path)


def save_one_image_in_images(images, path, generate_name, image_size=None, **kwargs):
    """
    Save one image in images
    :param images: Image
    :param generate_name: generate image name
    :param path: Save path
    :param image_size: Resize image size
    :param kwargs: Other parameters
    :return: None
    """
    # This is counter
    count = 0
    # Show image in images
    for i in images.cpu():
        grid = torchvision.utils.make_grid(tensor=i, **kwargs)
        image_array = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(obj=image_array)
        # Rename every images
        im.save(fp=os.path.join(path, f"{generate_name}_{count}.jpg"))
        if image_size is not None:
            logger.info(msg=f"Image is resizing {image_size}.")
            # Resize
            # TODO: Super-resolution algorithm replacement
            im = im.resize(size=(image_size, image_size), resample=Image.ANTIALIAS)
            im.save(fp=os.path.join(path, f"{generate_name}_{image_size}_{count}.jpg"))
        count += 1


class ImageFolderWithName(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        
        # Extract image name from the path
        image_name = path.split('/')[-1]  # Adjust this based on your path format
        
        return image, target, image_name

def get_dataset(args, dataset_path ,is_train, is_val=False, distributed=False):
    """
    Get dataset

    Automatically divide labels torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    If the dataset is as follow:
        dataset_path/class_1/image_1.jpg
        dataset_path/class_1/image_2.jpg
        ...
        dataset_path/class_2/image_1.jpg
        dataset_path/class_2/image_2.jpg
        ...

    'dataset_path' is the root directory of the dataset, 'class_1', 'class_2', etc. are different categories in
    the dataset, and each category contains several image files.

    Use the 'ImageFolder' class to conveniently load image datasets with this folder structure,
    and automatically assign corresponding labels to each image.

    You can specify the root directory where the dataset is located by passing the 'dataset_path' parameter,
    and perform operations such as image preprocessing and label conversion through other optional parameters.

    About Distributed Training:
    +------------------------+                     +-----------+
    |DistributedSampler      |                     |DataLoader |
    |                        |     2 indices       |           |
    |    Some strategy       +-------------------> |           |
    |                        |                     |           |
    |-------------+----------|                     |           |
                  ^                                |           |  4 data  +-------+
                  |                                |       -------------->+ train |
                1 | length                         |           |          +-------+
                  |                                |           |
    +-------------+----------+                     |           |
    |DataSet                 |                     |           |
    |        +---------+     |      3 Load         |           |
    |        |  Data   +-------------------------> |           |
    |        +---------+     |                     |           |
    |                        |                     |           |
    +------------------------+                     +-----------+

    :param args: Parameters
    :param distributed: Whether to distribute training
    :return: dataloader
    """
    transforms = torchvision.transforms.Compose([
        # Resize input size
        # torchvision.transforms.Resize(80), args.image_size + 1/4 * args.image_size
        # torchvision.transforms.Resize(size=int(args.image_size + args.image_size / 4)),
        torchvision.transforms.Resize(size=int(args.image_size)),
        # Random adjustment cropping
        # torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.0)),
        # To Tensor Format
        torchvision.transforms.ToTensor(),
        # For standardization, the mean and standard deviation are both 0.5
        torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # Load the folder data under the current path,
    # and automatically divide the labels according to the dataset under each file name
    if is_train :
        path_img = os.path.join(dataset_path,"train","img")
        path_lbl = os.path.join(dataset_path,"train","lbl")
        path_pred = os.path.join(dataset_path,"train","pred_sam")
        
    else :
        path_img = os.path.join(dataset_path,"test","img")
        path_lbl = os.path.join(dataset_path,"test","lbl")
        path_pred = os.path.join(dataset_path,"test","pred_sam")

    if is_val :
        path_img = os.path.join(dataset_path,"val","img")
        path_lbl = os.path.join(dataset_path,"val","lbl")
        path_pred = os.path.join(dataset_path,"val","pred_sam")
        
    dataset_img = ImageFolderWithName(root=path_img, transform=transforms)
    dataset_lbl = ImageFolderWithName(root=path_lbl, transform=transforms)
    dataset_pred_sam = ImageFolderWithName(root=path_pred, transform=transforms)
    dataloader_img = DataLoader(dataset=dataset_img, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    dataloader_lbl = DataLoader(dataset=dataset_lbl, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    dataloader_pred_sam = DataLoader(dataset=dataset_pred_sam, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
    return dataloader_img, dataloader_lbl, dataloader_pred_sam


def setup_logging(save_path, run_name):
    """
    Set log saving path
    :param save_path: Saving path
    :param run_name: Saving name
    :return: List of file paths
    """
    results_root_dir = save_path
    results_dir = os.path.join(save_path, run_name)
    results_vis_dir = os.path.join(save_path, run_name, "vis")
    results_tb_dir = os.path.join(save_path, run_name, "tensorboard")
    # Root folder
    os.makedirs(name=results_root_dir, exist_ok=True)
    # Saving folder
    os.makedirs(name=results_dir, exist_ok=True)
    # Visualization folder
    os.makedirs(name=results_vis_dir, exist_ok=True)
    # Visualization folder for Tensorboard
    os.makedirs(name=results_tb_dir, exist_ok=True)
    return [results_root_dir, results_dir, results_vis_dir, results_tb_dir]


def delete_files(path):
    """
    Clear files
    :param path: Path
    :return: None
    """
    if os.path.exists(path=path):
        if os.path.isfile(path=path):
            os.remove(path=path)
        else:
            shutil.rmtree(path=path)
        logger.info(msg=f"Folder '{path}' deleted.")
    else:
        logger.warning(msg=f"Folder '{path}' does not exist.")


def save_train_logging(arg, save_path):
    """
    Save train log
    :param arg: Argparse
    :param save_path: Save path
    :return: None
    """
    with open(file=f"{save_path}/train.log", mode="a") as f:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        f.write(f"{current_time}: {arg}\n")
    f.close()


def check_and_create_dir(path):
    """
    Check and create not exist folder
    :param path: Create path
    :return: None
    """
    logger.warning(msg=f"Folder '{path}' does not exist.")
    os.makedirs(name=path, exist_ok=True)
    logger.info(msg=f"Successfully create folder '{path}'.")
