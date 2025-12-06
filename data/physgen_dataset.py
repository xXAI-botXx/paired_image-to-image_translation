"""
PhysGen Dataset

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
from data.base_dataset import BaseDataset, get_transform

import os
import shutil

from PIL import Image
import cv2

# from datasets import load_dataset
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset

# import prime_printer as prime 
import img_phy_sim as ips

class PhysGenDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--is_train', action='store_true', help='Whether it is train or test.')
        parser.add_argument('--variation', type=str, default="sound_baseline", help='Decides which dataset to load: sound_baseline, sound_reflection, sound_diffraction, sound_combined.')
        parser.add_argument('--reflexion_channels', action='store_true', help='Whether to add channels with reflexion traces.')
        parser.add_argument('--reflexion_steps', type=int, default=36, help='Amount of reflexion beams.')
        parser.add_argument('--reflexions_as_channels', action='store_true', help='Whether to add channels with reflexion traces.')
        
        parser.set_defaults(max_dataset_size=float("inf"))  # specify dataset-specific default values
        return parser

    def __init__(self, opt, dataset, mode):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        try:
            self.resolution_512 = opt.resolution_512
        except AttributeError:
            self.resolution_512 = False

        # get data
        # self.dataset = load_dataset("mspitzna/physicsgen", name="sound_combined", trust_remote_code=True)
        self.dataset = dataset

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
        ])
        print(f"PhysGen Dataset for {mode} got created")
        self.mode = mode
        self.reflexion_channels = opt.reflexion_channels
        self.reflexion_steps = opt.reflexion_steps
        self.reflexions_as_channels = opt.reflexions_as_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # print(sample)
        # print(sample.keys())
        input_img = sample["osm"]  # PIL Image
        target_img = sample["soundmap"]  # PIL Image

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        # add raytracing
        if self.reflexion_channels:
            # first try to find the rays
            ray_path = os.path.join("./rays", self.mode, str(self.reflexion_steps), f"rays_[{str(idx)}].txt")
            if os.path.exists(ray_path):
                rays = ips.ray_tracing.open(path=ray_path)
            else:
                rays = ips.ray_tracing.trace_beams(rel_position=(0.5, 0.5),	
                                                    img_src=np.squeeze(input_img.cpu().numpy(), axis=0),	
                                                    directions_in_degree=ips.math.get_linear_degree_range(step_size=(self.reflexion_steps/360)*100),	
                                                    wall_values=[0],	
                                                    wall_thickness=0,	
                                                    img_border_also_collide=False,	
                                                    reflexion_order=3,	
                                                    should_scale_rays=True,	
                                                    should_scale_img=False)
            ray_img = ips.ray_tracing.draw_rays(rays,	
                                                detail_draw=False,	
                                                output_format='channels' if self.reflexions_as_channels else 'single_image',	
                                                img_background=None,	
                                                ray_value=[50, 100, 255],	
                                                ray_thickness=1,	
                                                img_shape=(512, 512),
                                                should_scale_rays_to_image=True,
                                                show_only_reflections=False)
            # (256, 256)
            ray_img = self.transform(ray_img)
            ray_img = ray_img.float()
            if ray_img.ndim == 2:
                ray_img = ray_img.unsqueeze(0)  # (1, H, W)

            # Merging with input image 
            if ray_img.shape[1:] == input_img.shape[1:]:
                input_img = torch.cat((input_img, ray_img), dim=0)
            else:
                raise ValueError(f"Ray image shape {ray_img.shape} does not match input image shape {input_img.shape}.")

        return input_img, target_img, idx




