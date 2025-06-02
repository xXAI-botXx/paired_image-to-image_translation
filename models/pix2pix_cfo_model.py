from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .pix2pix_cfo_sub_model import Pix2PixCFOSubModel
from data.extended_physgen_dataset import PhysGenDataset


class FusionHead(nn.Module):
    def __init__(self, input_channels, hidden_size=64):
        super(FusionHead, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, 1, kernel_size=1) 
        )
        self.loss = torch.nn.L1Loss()
        self.last_loss = float("inf")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def forward(self, x):
        return torch.sigmoid(self.fusion(x))

    def backward(self, target_, pred_):
        loss = self.loss(pred_, target_)
        self.last_loss = loss.cpu().detach()
        loss.backward()

class Pix2PixCFOModel(BaseModel):
    """ This class implements the pix2pix complex focus only model, for learning a mapping from input images to output images given paired data.

    This model is a special model, which handles data by them own.

    Idea is Residual Learning:
    Pix2Pix Model 1: OSM -> Baseline Propagation
    Pix2Pix Model 2: OSM -> Only Complex (Reflection or Diffraction - Baseline)
    Fusion Head: Pix2Pix Model 1 + Pix2Pix Model 2 -> Reflection or Diffraction

    So we have 3 different models with each other input-output data. 

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_second', type=float, default=100.0, help='weight for Second loss (L1)')
            parser.add_argument('--wgangp', action='store_true', help='Should use WGAN-GP')
            parser.add_argument('--use_cfg_loss', action='store_true', help='Whether to use a special complex focus only loss.')
            # parser.add_argument("--variation", help="Dataset variant: sound_baseline, sound_reflection, sound_diffraction, sound_combined")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.opt = opt

        base_opt = deepcopy(opt)
        base_opt.use_cfg_loss = False
        self.base_model = Pix2PixCFOSubModel(base_opt)
        self.netbase_model_g = self.base_model.netG
        self.netbase_model_d = self.base_model.netD
        
        complex_opt = deepcopy(opt)
        complex_opt.use_cfg_loss = True
        self.complex_model = self.netcomplex_model = Pix2PixCFOSubModel(complex_opt)
        self.netcomplex_model_g = self.complex_model.netG
        self.netcomplex_model_d = self.complex_model.netD

        self.fusion_head = self.netfusion_head = self.netfusion_head = FusionHead(input_channels=2)

        self.optimizers = [*self.base_model.optimizers, *self.complex_model.optimizers, self.fusion_head.optimizer]
        self.model_names = ['base_model_g', 'base_model_d', 'complex_model_g', 'complex_model_d']  # 'fusion_head'
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.loss_base_model_g = self.base_model.loss_G_GAN
        self.loss_base_model_second = self.base_model.loss_second
        self.loss_base_model_d = self.base_model.loss_D_real
        self.loss_complex_model_g = self.complex_model.loss_G_GAN
        self.loss_complex_model_second = self.complex_model.loss_second
        self.loss_complex_model_d = self.complex_model.loss_D_real
        self.loss_fusion = self.fusion_head.last_loss
        self.loss_names = ['base_model_g',
                           'base_model_second',
                           'base_model_d',
                           'complex_model_g',
                           'complex_model_second',
                           'complex_model_d',
                           'fusion']

        self.isTrain = opt.isTrain
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.train_dataset_base = PhysGenDataset(mode='train', variation="sound_baseline", input_type="osm", output_type="standard")
            self.val_dataset_base = PhysGenDataset(mode='validation', variation="sound_baseline", input_type="osm", output_type="standard")

            self.train_dataset_complex = PhysGenDataset(mode='train', variation=opt.variation, input_type="osm", output_type="complex_only")
            self.val_dataset_complex = PhysGenDataset(mode='validation', variation=opt.variation, input_type="osm", output_type="complex_only")

            self.train_dataset_fusion = PhysGenDataset(mode='train', variation=opt.variation, input_type="osm", output_type="standard")
            self.val_dataset_fusion = PhysGenDataset(mode='validation', variation=opt.variation, input_type="osm", output_type="standard")
            self.datasets = [(self.train_dataset_base, self.val_dataset_base), (self.train_dataset_complex, self.val_dataset_complex), (self.train_dataset_fusion, self.val_dataset_fusion)]

        self.epochs = opt.n_epochs
        self.train_pix2pix_epochs = int(self.epochs*0.8)
        self.current_epoch = 0
        self.should_validate = False
        self.data_idx_train = 0
        self.data_idx_val = 0
            

    def set_input(self, input_):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.base_model.set_input(input_)
        self.complex_model.set_input(input_)
        
        self.real_A = self.base_model.real_A
        self.real_B = self.base_model.real_B

    def set_to_validation(self):
        self.should_validate = True
        self.data_idx_train = 0
        self.data_idx_val = 0

    def set_to_train(self):
        self.should_validate = False
        self.data_idx_train = 0
        self.data_idx_val = 0

    def set_current_epoch(self, epoch):
        new_epoch = self.current_epoch != epoch
        self.current_epoch = epoch

    def forward(self, model_idx=0):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.forward_and_return(model_idx=model_idx)

    def forward_and_return(self, model_idx=0):
        """Run forward pass and returns the output"""
        
        if self.isTrain:
            if self.should_validate:
                base_data = to_device(self.datasets[0][1][self.data_idx_val])
                complex_data = to_device(self.datasets[1][1][self.data_idx_val])
                _, target_, idx_ = to_device(self.datasets[2][1][self.data_idx_val])

                base_pred = self.base_model.forward_and_return(*base_data).unsqueeze(1)
                complex_pred = self.complex_model.forward_and_return(*complex_data).unsqueeze(1)
                
                combined = torch.cat([base_x, complex_x], dim=1)
                pred = self.fusion_head(combined)
                if len(pred.shape) == 4:
                    pred = pred.squeeze(1)

                self.data_idx_val += 1
            else:
                base_data = to_device(self.datasets[0][0][self.data_idx_train])
                complex_data = to_device(self.datasets[1][0][self.data_idx_train])
                _, target_, idx_ = to_device(self.datasets[2][0][self.data_idx_train])
                
                if model_idx == 0:
                    pred = self.base_model.forward_and_return(base_data[0], base_data[1]).unsqueeze(1)
                elif model_idx == 1:
                    pred = self.complex_model.forward_and_return(complex_data[0], complex_data[1]).unsqueeze(1)
                else:
                    base_pred = self.base_model.forward_and_return(base_data[0], base_data[1]).unsqueeze(1)
                    complex_pred = self.complex_model.forward_and_return(complex_data[0], complex_data[1]).unsqueeze(1)
                    
                    combined = torch.cat([base_x, complex_x], dim=1)
                    pred = self.fusion_head(combined)
                    if len(pred.shape) == 4:
                        pred = pred.squeeze(1)

                self.data_idx_train += 1
        else:
            base_pred = self.base_model(self.real_A).unsqueeze(1)
            complex_pred = self.complex_model(self.real_A).unsqueeze(1)
            
            combined = torch.cat([base_x, complex_x], dim=1)
            pred = self.fusion_head(combined)
            if len(pred.shape) == 4:
                pred = pred.squeeze(1)

        # fake_B = pred
        # if fake_B.dim() == 5:
        #     fake_B = fake_B.squeeze(0).squeeze(0)
        # elif fake_B.dim() == 4:
        #     fake_B = fake_B.squeeze(0)
        # elif fake_B.dim() == 2:
        #     fake_B = fake_B.unsqueeze(0)
        self.fake_B = pred
        return pred

    def adjust_image_shapes(self):
        if self.fake_B.dim() == 5:
            self.fake_B = self.fake_B.squeeze(0).squeeze(0)
        elif self.fake_B.dim() == 4:
            self.fake_B = self.fake_B.squeeze(0)
        elif self.fake_B.dim() == 2:
            self.fake_B = self.fake_B.unsqueeze(0)
        
        if self.real_B.dim() == 5:
            self.real_B = self.real_B.squeeze(0).squeeze(0)
        elif self.real_B.dim() == 4:
            self.real_B = self.real_B.squeeze(0)
        elif self.real_B.dim() == 2:
            self.real_B = self.real_B.unsqueeze(0)

        if self.real_A.dim() == 5:
            self.real_A = self.real_A.squeeze(0).squeeze(0)
        elif self.real_A.dim() == 4:
            self.real_A = self.real_A.squeeze(0)
        elif self.real_A.dim() == 2:
            self.real_A = self.real_A.unsqueeze(0)

    def update_loss(self):
        self.loss_base_model_g = self.base_model.loss_G_GAN
        self.loss_base_model_second = self.base_model.loss_second
        self.loss_base_model_d = self.base_model.loss_D_real
        self.loss_complex_model_g = self.complex_model.loss_G_GAN
        self.loss_complex_model_second = self.complex_model.loss_second
        self.loss_complex_model_d = self.complex_model.loss_D_real
        self.loss_fusion = self.fusion_head.last_loss

    def optimize_parameters(self):
        """
        For every model:
        1. model.forward
        2. model.set_requires_grad(model.netD, True)
        3. model.optimizer_D.zero_grad()
        4. model.backward_D()
        5. model.optimizer_D.step()
        6. model.set_requires_grad(model.netD, False)
        7. model.optimizer_G.zero_grad()
        8. model.backward_G()
        9. model.optimizer_G.step()
        """
        base_data = to_device(self.datasets[0][0][self.data_idx_train])
        complex_data = to_device(self.datasets[1][0][self.data_idx_train])
        _, target_, idx_ = to_device(self.datasets[2][0][self.data_idx_train])

        if self.current_epoch <= self.train_pix2pix_epochs:
            # Basline
            pred_ = self.forward_and_return(model_idx=0)
            self.base_model.optimize_parameters(base_data[0], base_data[1], pred_)

            # Complex
            pred_ = self.forward_and_return(model_idx=1)
            self.base_model.optimize_parameters(complex_data[0], complex_data[1], pred_)
        else:
            # Fusion
            self.fusion_head.optimizer.zero_grad()
            pred_ = self.forward_and_return(model_idx=2)
            self.fusion_head.backward(target_, pred_)
            self.fusion_head.optimizer.step()

        self.data_idx_train += 0
        self.update_loss()
        # self.adjust_image_shapes()


def to_device(dataset):
    # Input: [Tensor(), Tensor(), int]
    if len(dataset) != 3:
        raise ValurError("Expected dataset to be a list of 3 values")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [dataset[0].to(device), dataset[1].to(device), dataset[2]]





