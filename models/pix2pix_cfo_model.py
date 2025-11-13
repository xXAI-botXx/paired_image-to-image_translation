from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .pix2pix_cfo_sub_model import Pix2PixCFOSubModel, scale


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
            parser.add_argument('--calc_weight_map_for_cfg_loss', action='store_true', help='Whether to use s weight map for the complex loss.')
        parser.add_argument('--reducing_cpu_bottleneck_over_gpu_memory', action='store_true', help='Whether to load all data to GPU or seperately load them to reduce GPU memory usage.')
        parser.add_argument('--using_fusion_head', action='store_true', help='Whether to use the CNN Fusion Head for combining or the math calc formular.')
        parser.add_argument('--scale_complex_part', action='store_true', help='Whether to upscale (downscaling on inference) the values to make the value ranges bigger and more easy to learn.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY = opt.reducing_cpu_bottleneck_over_gpu_memory
        self.USE_FUSION_MODEL = opt.using_fusion_head # if hasattr(opt, 'using_fusion_head') else False
        self.opt = opt
        self.isTrain = opt.isTrain

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_opt = deepcopy(opt)
        base_opt.use_cfg_loss = False
        self.base_model = Pix2PixCFOSubModel(base_opt, is_base_model=True)
        self.netbase_model_g = self.base_model.netG
        if self.isTrain:
            self.netbase_model_d = self.base_model.netD
        
        complex_opt = deepcopy(opt)
        complex_opt.use_cfg_loss = True
        self.complex_model = Pix2PixCFOSubModel(complex_opt, is_base_model=False)
        self.netcomplex_model_g = self.complex_model.netG
        if self.isTrain:
            self.netcomplex_model_d = self.complex_model.netD

        self.fusion_head = self.netfusion_head = FusionHead(input_channels=2)
        if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
            self.fusion_head = self.fusion_head.to(self.device)

        if self.isTrain:
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

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.epochs = opt.n_epochs
            self.train_pix2pix_epochs = int(self.epochs*0.8)
            self.debug_print = True
        self.current_epoch = 0
        self.should_validate = False
            

    def set_input(self, input_):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.isTrain:
            # move to GPU if flag is set
            if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                # input_[0][0] = to_device(input_[0][0], self.device)
                # input_[0][1] = to_device(input_[0][1], self.device)
                # input_[1][0] = to_device(input_[1][0], self.device)
                # input_[1][1] = to_device(input_[1][1], self.device)
                input_ = ((to_device(input_[0][0], self.device), to_device(input_[0][1], self.device)), \
                          (to_device(input_[1][0], self.device), to_device(input_[1][1], self.device)), \
                          (input_[2][0], input_[2][1]), \
                          input_[3])

            (input_base_inputs, input_base_targets), \
            (input_complex_inputs, input_complex_targets), \
            (_, _), \
            idx = input_

            self.current_data = input_

            input_ = [input_base_inputs, input_base_targets, idx]
            input_complex = [input_complex_inputs, input_complex_targets, idx]

            self.base_model.set_input(input_)
            self.complex_model.set_input(input_complex)
        else:
            input_ = [input_[0] if input_[0].ndim == 4 else input_[0].unsqueeze(0), \
                      input_[1] if input_[1].ndim == 4 else input_[1].unsqueeze(0), \
                      input_[2]]
            
            # move to GPU if flag is set
            if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                input_[0] = to_device(input_[0], self.device)
                input_[1] = to_device(input_[1], self.device)

            self.current_data = input_
            # input_ = [inputs, targets, idx]

            self.base_model.set_input(input_)
            self.complex_model.set_input(input_)
        
        self.real_A = self.base_model.real_A
        self.real_B = self.base_model.real_B

        self.image_names_dict = OrderedDict()
        self.image_names_dict[f'real_A'] = input_[0] if len(input_[0].shape) == 4 else input_[0].unsqueeze(0)
        self.image_names_dict[f'fake_B'] = None 
        self.image_names_dict[f'real_B'] = input_[1] if len(input_[1].shape) == 4 else input_[1].unsqueeze(0)

        self.image_paths = ["./cache_physgen/" + f"building_{input_[2]}.png" if self.opt.direction == 'AtoB' else f"{input_[2]}_LAEQ.png"]

    def set_to_validation(self):
        self.should_validate = True
        # self.train_loader.current_index = 0
        # self.val_loader.current_index = 0

    def set_to_train(self):
        self.should_validate = False
        # self.train_loader.current_index = 0
        # self.val_loader.current_index = 0

    def set_current_epoch(self, epoch):
        new_epoch = self.current_epoch != epoch
        self.current_epoch = epoch

    def clean(self):
        self.base_model.clean()
        self.complex_model.clean()

    def forward(self, model_idx=0):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.forward_and_return(model_idx=model_idx)

    def forward_and_return(self, model_idx=0, should_take_last=False):
        """
        Run forward pass and returns the output
        
        Have to choose between CPU bottleneck (moving dataets between devices)
        or GPU memory bottleneck (keeping datasets on GPU).
        """
        
        if self.isTrain:
            # (train_base_inputs, train_base_targets), \
            # (train_complex_inputs, train_complex_targets), \
            # (train_fusion_inputs, train_fusion_targets), idx = self.train_loader.get_next() if not should_take_last else self.train_loader.get_last()

            # (val_base_inputs, val_base_targets), \
            # (val_complex_inputs, val_complex_targets), \
            # (val_fusion_inputs, val_fusion_targets), idx = self.val_loader.get_next() if not should_take_last else self.val_loader.get_last()

            # if self.debug_print:
            #     print(f"\n[DEBUG] Shapes in forward_and_return:\
            #             \n    - Base Input {train_base_inputs.shape}\
            #             \n    - Base Target {train_base_targets.shape} \
            #             \n    - Complex Input {train_complex_inputs.shape} \
            #             \n    - Complex Target {train_complex_targets.shape}\n")
            #     self.debug_print = False

            if self.should_validate:
                (val_base_inputs, val_base_targets), \
                (val_complex_inputs, val_complex_targets), \
                (val_fusion_inputs, val_fusion_targets), idx = self.current_data

                if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                    base_data = (val_base_inputs, val_base_targets)
                else:
                    base_data = (to_device(val_base_inputs, self.device), to_device(val_base_targets, self.device))
                base_pred = self.base_model.forward_and_return(*base_data)
                base_pred = base_pred if base_pred.dim() == 4 else base_pred.unsqueeze(1)
                if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                    base_data[0].cpu().detach()
                    base_data[1].cpu().detach()

                if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                    complex_data = (val_complex_inputs, val_complex_targets)
                else:
                    complex_data = (to_device(val_complex_inputs, self.device), to_device(val_complex_targets, self.device))
                complex_pred = self.complex_model.forward_and_return(*complex_data, should_scale=True)
                complex_pred = complex_pred if complex_pred.dim() == 4 else complex_pred.unsqueeze(1)
                if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                    complex_data[0].cpu().detach()
                    complex_data[1].cpu().detach()
                
                combined = torch.cat([base_pred, complex_pred], dim=1)
                pred = self.fusion_head(combined)
                if len(pred.shape) == 4:
                    pred = pred.squeeze(1)
            else:
                (train_base_inputs, train_base_targets), \
                (train_complex_inputs, train_complex_targets), \
                (train_fusion_inputs, train_fusion_targets), idx = self.current_data

                if model_idx == 0:
                    if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        base_data = (train_base_inputs, train_base_targets)
                    else:
                        base_data = (to_device(train_base_inputs, self.device), to_device(train_base_targets, self.device))
                    pred = self.base_model.forward_and_return(base_data[0], base_data[1])
                    pred = pred if pred.dim() == 4 else pred.unsqueeze(1)
                    if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        base_data[0].cpu().detach()
                        base_data[1].cpu().detach()
                elif model_idx == 1:
                    if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        complex_data = (train_complex_inputs, train_complex_targets)
                    else:
                        complex_data = (to_device(train_complex_inputs, self.device), to_device(train_complex_targets, self.device))
                    pred = self.complex_model.forward_and_return(complex_data[0], complex_data[1], False)
                    pred = pred if pred.dim() == 4 else pred.unsqueeze(1)
                    if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        complex_data[0].cpu().detach()
                        complex_data[1].cpu().detach()
                else:
                    if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        base_data = (train_base_inputs, train_base_targets)
                    else:
                        base_data = (to_device(train_base_inputs, self.device), to_device(train_base_targets, self.device))
                    base_pred = self.base_model.forward_and_return(base_data[0], base_data[1])
                    base_pred = base_pred if base_pred.dim() == 4 else base_pred.unsqueeze(1)
                    if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        base_data[0].cpu().detach()
                        base_data[1].cpu().detach()

                    if self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        complex_data = (train_complex_inputs, train_complex_targets)
                    else:
                        complex_data = (to_device(train_complex_inputs, self.device), to_device(train_complex_targets, self.device))
                    complex_pred = self.complex_model.forward_and_return(complex_data[0], complex_data[1], should_scale=True)
                    complex_pred = complex_pred if complex_pred.dim() == 4 else complex_pred.unsqueeze(1)
                    if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        complex_data[0].cpu().detach()
                        complex_data[1].cpu().detach()
                    
                    combined = torch.cat([base_pred, complex_pred], dim=1)
                    if not self.REDUCING_CPU_BOTTLENECK_OVER_GPU_MEMORY:
                        self.fusion_head = self.fusion_head.to(combined.device)
                    pred = self.fusion_head(combined)
                    if len(pred.shape) == 4:
                        pred = pred.squeeze(1)

        else:
            base_pred = self.base_model(self.real_A)
            complex_pred = self.complex_model(self.real_A, should_scale=True)
            
            if self.USE_FUSION_MODEL:
                base_pred = base_pred if base_pred.dim() == 4 else base_pred.unsqueeze(1)
                complex_pred = complex_pred if complex_pred.dim() == 4 else complex_pred.unsqueeze(1)
                combined = torch.cat([base_pred, complex_pred], dim=1)
                pred = self.fusion_head(combined)
            else:
                # formular:
                #   complex = (target - base) * -2
                #   complex*(-0.5) = target - base
                #   target = (complex*(-0.5)) + base
                pred = (complex_pred*(-0.5)) + base_pred

            if len(pred.shape) == 4:
                pred = pred.squeeze(1)

        self.fake_B = pred
        self.image_names_dict['fake_B'] = self.fake_B if len(self.fake_B.shape) == 4 else self.fake_B.unsqueeze(0)
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
        (train_base_inputs, train_base_targets), \
        (train_complex_inputs, train_complex_targets), \
        (train_fusion_inputs, train_fusion_targets), idx = self.current_data

        if self.current_epoch <= self.train_pix2pix_epochs:
            # Basline
            pred_ = self.forward_and_return(model_idx=0, should_take_last=True)
            base_data = (to_device(train_base_inputs, self.device), to_device(train_base_targets, self.device))
            self.base_model.optimize_parameters(base_data[0], base_data[1], pred_)
            base_data[0].cpu().detach()
            base_data[1].cpu().detach()
            # print(f"\nBaseline Prediction Output:\n    - min = {pred_.min().item()}\n    - max = {pred_.max().item()}\n    - mean = {pred_.mean().item()}\n    - var = {pred_.var().item()}\n    - nan = {torch.isnan(pred_).any()}")

            # Complex
            pred_ = self.forward_and_return(model_idx=1, should_take_last=True)
            complex_data = (to_device(train_complex_inputs, self.device), to_device(train_complex_targets, self.device))
            self.complex_model.optimize_parameters(complex_data[0], complex_data[1], pred_)
            complex_data[0].cpu().detach()
            complex_data[1].cpu().detach()
            # print(f"\nComplex Prediction Output:\n    - min = {pred_.min().item()}\n    - max = {pred_.max().item()}\n    - mean = {pred_.mean().item()}\n    - var = {pred_.var().item()}\n    - nan = {torch.isnan(pred_).any()}")
        else:
            # Fusion
            self.fusion_head.optimizer.zero_grad()
            pred_ = self.forward_and_return(model_idx=2, should_take_last=True)
            train_fusion_targets = to_device(train_fusion_targets, self.device)
            self.fusion_head.backward(train_fusion_targets, pred_)
            train_fusion_targets.cpu().detach()
            self.fusion_head.optimizer.step()
            # print(f"\nFusion Prediction Output:\n    - min = {pred_.min().item()}\n    - max = {pred_.max().item()}\n    - mean = {pred_.mean().item()}\n    - var = {pred_.var().item()}\n    - nan = {torch.isnan(pred_).any()}")

        self.update_loss()
        # self.adjust_image_shapes()


def to_device(dataset, device=None):
    # Input: [Tensor(), Tensor(), int]
    # if len(dataset) != 2:
    #     raise ValueError("Expected dataset to be a list of 2 values")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(dataset, (list, tuple)):
        return type(dataset)(item.to(device) for item in dataset)
    elif isinstance(dataset, torch.Tensor):
        return dataset.to(device)
    else:
        return dataset # [dataset[0].to(device), dataset[1].to(device)]


def get_image_paths(self):
    pass





