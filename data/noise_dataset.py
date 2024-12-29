"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import os
from PIL import Image
import torchvision.transforms as transforms

# from data.image_folder import make_dataset
# from PIL import Image


class NoiseDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
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
        # get the image paths of your dataset;
        self.building_path = os.path.join(self.root, "buildings")  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        self.noise_maps_path = os.path.join(self.root, "interpolated")

        self.resolution_512 = opt.resolution_512
        self.building_paths = sorted(os.listdir(self.building_path))
        self.noise_maps_paths = sorted(os.listdir(self.noise_maps_path))
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)
        print("NOISEDATASET object created")


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # get the corresponding index of the noise map list depending on which resolution we want to train 
        noise_maps_index = 2 * index if not self.resolution_512 else (2 * index) + 1 

        # get paths
        path = 'NOT IMPLEMENTED FOR NOISE DATASET'  # needs to be a string
        path_A = os.path.join(self.building_path, f"buildings_{index}.png")
        #print(path_A)
        if self.resolution_512:
            path_B = os.path.join(self.noise_maps_path, f"{index}_LEQ_512.png")
        else:
            path_B = os.path.join(self.noise_maps_path, f"{index}_LEQ_256.png")
        #print(path_B)

        # load image and gt
        data_A =  Image.open(path_A).convert('L')
        data_B = Image.open(path_B).convert('L')

        # convert to torch tensor
        to_tensor = transforms.ToTensor()
        data_A = to_tensor(data_A)
        data_B = to_tensor(data_B)

        return {'A': data_A, 'B': data_B, 'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        """Return the total number of images."""
        return len(self.building_paths)
