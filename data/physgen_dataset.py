"""
PhysGen Dataset

See:
- https://huggingface.co/datasets/mspitzna/physicsgen
- https://arxiv.org/abs/2503.05333
- https://github.com/physicsgen/physicsgen
"""
from data.base_dataset import BaseDataset, get_transform

import os
from PIL import Image

# from datasets import load_dataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms


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
        parser.add_argument('--is_train', action='store_true', help='Whether it is train or test.')
        parser.add_argument('--variation', type=str, default="sound_baseline", help='Decides which dataset to load: sound_baseline, sound_reflection, sound_diffraction, sound_combined.')
        
        parser.set_defaults(max_dataset_size=float("inf"))  # specify dataset-specific default values
        return parser

    def __init__(self, opt, dataset):
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
        self.resolution_512 = opt.resolution_512

        # get data
        # self.dataset = load_dataset("mspitzna/physicsgen", name="sound_combined", trust_remote_code=True)
        self.dataset = dataset

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] PIL image to [0,1] FloatTensor
        ])
        print(f"PhysGen Dataset for {'train' if opt.is_train else 'test'} got created")

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

        return input_img, target_img





