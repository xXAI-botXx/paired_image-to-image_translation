"""
PhysGen Dataset

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
import os
from PIL import Image

from datasets import load_dataset

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
from torchvision import transforms

import img_phy_sim as ips

# def resize_to_divisible_by_14(image: np.ndarray) -> np.ndarray:
#     """
#     Resize an image to the next smaller width and height that is divisible by 14.
    
#     Parameters:
#         image (np.ndarray): Input image (H x W x C or H x W).
        
#     Returns:
#         np.ndarray: Resized image.
#     """
#     original_channels, original_height, original_width = image.shape[:]

#     new_width = original_width - (original_width % 14)
#     new_height = original_height - (original_height % 14)

#     resized_image = cv2.resize(image, (original_channels, new_width, new_height), interpolation=cv2.INTER_AREA)
#     return resized_image

def resize_tensor_to_divisible_by_14(tensor: torch.Tensor) -> torch.Tensor:
    """
    Resize a tensor to the next smaller (H, W) divisible by 14.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (B, C, H, W)
    
    Returns:
        torch.Tensor: Resized tensor
    """
    if tensor.dim() == 3:
        c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    
    elif tensor.dim() == 4:
        b, c, h, w = tensor.shape
        new_h = h - (h % 14)
        new_w = w - (w % 14)
        return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    else:
        raise ValueError("Tensor must be 3D (C, H, W) or 4D (B, C, H, W)")


class PhysGenDataset(Dataset):

    def __init__(self, variation="sound_baseline", mode="train", input_type="osm", output_type="standard", 
                 reflexion_channels=False, reflexion_steps=36, reflexions_as_channels=False):
        """
        Loads PhysGen Dataset.

        Parameters:
        - variation : str
            Chooses the used dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined.
        - mode : str
            Can be "train", "test", "eval".
        - input_type : str
            Defines the used Input -> "osm", "base_simulation"
        - output_type : str
            Defines the Output -> "standard", "complex_only"
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # get data
        self.dataset = load_dataset("mspitzna/physicsgen", name=variation, trust_remote_code=True)
        self.dataset = self.dataset[mode]

        self.mode = mode
        # self.is_train = mode == "train"
        
        self.input_type = input_type
        self.output_type = output_type
        if self.input_type == "base_simulation" or self.output_type == "complex_only":
            self.basesimulation_dataset = load_dataset("mspitzna/physicsgen", name="sound_baseline", trust_remote_code=True)
            self.basesimulation_dataset = self.basesimulation_dataset[mode]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
        ])
        print(f"PhysGen ({variation}) Dataset for {mode} got created")
        self.reflexion_channels = reflexion_channels
        self.reflexion_steps = reflexion_steps
        self.reflexions_as_channels = reflexions_as_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # print(sample)
        # print(sample.keys())
        if self.input_type == "base_simulation":
            input_img = self.basesimulation_dataset[idx]["soundmap"]
        else:
            input_img = sample["osm"]  # PIL Image
        target_img = sample["soundmap"]  # PIL Image

        input_img = self.transform(input_img)
        target_img = self.transform(target_img)

        # Fix real image size 512x512 > 256x256
        input_img = F.interpolate(input_img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        input_img = input_img.squeeze(0)
        # target_img = target_img.unsqueeze(0)

        # change size
        # input_img = resize_tensor_to_divisible_by_14(input_img)
        # target_img = resize_tensor_to_divisible_by_14(target_img)

        # add fake rgb
        # if input_img.shape[0] == 1:  # shape (B, 1, H, W)
        #     input_img = input_img.repeat(3, 1, 1)  # make it (B, 3, H, W)

        if self.output_type == "complex_only":
            # base_simulation_img = resize_tensor_to_divisible_by_14(self.transform(self.basesimulation_dataset[idx]["soundmap"]))
            base_simulation_img = self.transform(self.basesimulation_dataset[idx]["soundmap"])
            # target_img = torch.abs(target_img[0] - base_simulation_img[0])
            target_img = target_img[0] - base_simulation_img[0]
            target_img = target_img.unsqueeze(0)
            target_img *= -2
            # target_img = torch.log1p(target_img)
            # target_img *= 100

        # reflexions
        if self.reflexion_channels:
            height, width = np.squeeze(input_img.cpu().numpy(), axis=0).shape
            ray_path = os.path.join("./rays", self.mode, str(self.reflexion_steps), f"rays_[{str(idx)}].txt")
            if os.path.exists(ray_path):
                rays = ips.ray_tracing.open(path=ray_path)
            else:
                # raise Exception("Ray file not found. Please generate ray files before using reflexion channels.")
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
                                                img_shape=(height, width),
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





