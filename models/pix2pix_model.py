import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import kornia

from .base_model import BaseModel
from . import networks
from .pix2pix_cfo_sub_model import WeightedCombinedLoss, calc_weight_map



def masked_l1_loss(prediction, target, mask):
    diff = torch.abs(prediction - target) * mask
    return diff.sum() / (mask.sum() + 1e-8)



def shifted_mask(batch_size, height, width, device, region_size=(16, 16), epoch=0, shift_every_n_epochs=10, max_epoch=80):
    # check if max is reached -> 20% of max epoch
    limit_usage_epoch = int(max_epoch * 0.2)
    if epoch >= limit_usage_epoch:
        return torch.ones((batch_size, 1, height, width), device=device) 

    # init with full mask
    mask = torch.zeros((batch_size, 1, height, width), device=device)

    # amount of regions per row and column
    n_rows = height // region_size[0]
    n_cols = width // region_size[1]
    total_regions = n_rows * n_cols

    # get region which should be active
    shift_index = (epoch // shift_every_n_epochs) % total_regions
    row_idx = shift_index // n_cols
    col_idx = shift_index % n_cols

    # scale to the real pixel size
    row_start = row_idx * region_size[0]
    col_start = col_idx * region_size[1]

    # masked regions set to 1
    mask[:, :, row_start:row_start+region_size[0],
               col_start:col_start+region_size[1]] = 1

    return mask



class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

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
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--wgangp', action='store_true', help='Should use WGAN-GP')
            parser.add_argument('--masked', action='store_true', help='Should mask with the target and threshold at 0')
            parser.add_argument('--post_masked', action='store_true', help='Should mask with the target and threshold at 50%')
            parser.add_argument('--use_weighted_loss', action='store_true', help='Should use weighted loss or standard l1-loss.')
            parser.add_argument('--calc_weight_map_for_weighted_loss', action='store_true', help='Whether to use weight map for the weighted loss.')
            parser.add_argument('--activate_gan_mid_refinement', action='store_true', help='If activated the gan will be purely used as loss at 20 to 80% of the training process.')
        parser.add_argument('--only_reflexions', action='store_true', help='Whether to use only the reflexions as input if using reflexions.')

            # print("modify: default weighted_loss =", parser.get_default('use_weighted_loss'))

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.masked = opt.masked
            self.post_masked = opt.post_masked
            self.use_weighted_loss = opt.use_weighted_loss
            self.calc_weight_map_for_weighted_loss = opt.calc_weight_map_for_weighted_loss
            self.first_loss_pass = True
            self.train_mask_area = True

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.epochs = opt.n_epochs
            self.mask = shifted_mask(batch_size=opt.batch_size, height=256, width=256, device=self.device, region_size=(16, 16), epoch=0, shift_every_n_epochs=10)
            self.batch_size = opt.batch_size
            self.activate_gan_mid_refinement = opt.activate_gan_mid_refinement
        
            self.lambda_GAN = self.opt.lambda_GAN
            self.lambda_L1 = self.opt.lambda_L1
        self.epochs_with_gan = 0
        self.forward_passes = 0
        self.current_epoch = 0

        if self.isTrain and self.activate_gan_mid_refinement:
            self.original_lambda_GAN = self.lambda_GAN
            self.original_lambda_L1 = self.lambda_L1

        self.only_reflexions = opt.only_reflexions

        # pix2pix_1_0_residual_few_bui_masked_weight_loss_2
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=0.0
        #                         )

        # pix2pix_1_0_residual_few_bui_masked_weight_loss_3
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=10.0, 
        #                                     weight_grad=10.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=10.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=10.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_4
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=10.0, 
        #                                     weight_grad=100.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=100.0,
        #                                     weight_l1=1000.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_5
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=10.0, 
        #                                     weight_ssim=5.0,
        #                                     weight_edge_aware=10.0,
        #                                     weight_l1=1000.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_6
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=1.0, 
        #                                     weight_grad=1.0, 
        #                                     weight_ssim=1.0,
        #                                     weight_edge_aware=1.0,
        #                                     weight_l1=1.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=1.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_7
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=1000.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_8
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=10.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=10.0,
        #                                     weight_l1=10.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # pix2pix_1_0_residual_few_bui_masked_weight_loss_9
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=10.0, 
        #                                     weight_grad=10.0, 
        #                                     weight_ssim=50.0,
        #                                     weight_edge_aware=10.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_10 -> good
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=100.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=100.0,
        #                                     weight_l1=10.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_11
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=100.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=100.0,
        #                                     weight_l1=10.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_12
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=100.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=100.0,
        #                                     weight_l1=10.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_13 -> good
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_14
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=50.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_15
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_16
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=10.0
        #                         )
        
        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_17
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=10.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_18
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=10.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=10.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_19
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=10.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=10.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_20 - 24
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_25
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=0.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_26
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=10.0
        #                         )

        # # pix2pix_1_0_residual_few_bui_masked_weight_loss_27
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=100.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=1.0,
        #                                     weight_var=1.0,
        #                                     weight_range=1.0,
        #                                     weight_blur=1.0
        #                         )

        # pix2pix_1_0_residual_few_bui_masked_engineered
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=1000.0,
        #                                     weight_l1=0.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=10.0
        #                         )

        # pix2pix_1_0_residual_few_bui_weight_loss_wgangp
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=1000.0, 
        #                                     weight_ssim=50.0,
        #                                     weight_edge_aware=1000.0,
        #                                     weight_l1=1000.0,
        #                                     weight_var=10.0,
        #                                     weight_range=10.0,
        #                                     weight_blur=100.0
        #                         )

        # pix2pix_1_0_residual_few_bui_weight_loss_wgangp_l1
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=0.0,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=100.0,
        #                                     weight_var=0.0,
        #                                     weight_range=0.0,
        #                                     weight_blur=0.0
        #                         )
        # self.lambda_GAN = 1000.0

        # pix2pix reflexion channels
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     silog_lambda=0.5, 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=0.0, 
        #                                     weight_ssim=0.1,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=1.0,
        #                                     weight_var=0.01,
        #                                     weight_range=0.01,
        #                                     weight_blur=0.1)
        
        # self.weighted_loss = WeightedCombinedLoss( 
        #                                     silog_lambda=0.5, 
        #                                     weight_silog=0.0, 
        #                                     weight_grad=1.0, 
        #                                     weight_ssim=0.3,
        #                                     weight_edge_aware=0.0,
        #                                     weight_l1=0.5,
        #                                     weight_var=0.1,
        #                                     weight_range=0.00,
        #                                     weight_blur=0.0)

        self.weighted_loss = WeightedCombinedLoss( 
                                            silog_lambda=0.5, 
                                            weight_silog=0.0, 
                                            weight_grad=1.0, 
                                            weight_ssim=0.3,
                                            weight_edge_aware=0.0,
                                            weight_l1=1.0,
                                            weight_var=0.05,
                                            weight_range=0.0,
                                            weight_blur=0.0)
        
        if hasattr(self, "use_weighted_loss") and not self.use_weighted_loss:
            self.weighted_loss = None

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.dataset_mode.lower() == "physgen":
            self.real_A = input[0].to(self.device)
            # Fix real image size 512x512 > 256x256
            self.real_A = F.interpolate(self.real_A.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            # self.real_A = self.real_A.squeeze(0)

            self.real_A = self.shrink_to_second_channel(self.real_A)

            self.real_B = input[1].to(self.device)
            self.real_B = self.real_B.unsqueeze(0)
            
            from collections import OrderedDict
            self.image_names_dict = OrderedDict()
            self.image_names_dict[f'real_A'] = input[0] if len(input[0].shape) == 4 else input[0].unsqueeze(0)
            self.image_names_dict[f'fake_B'] = None 
            self.image_names_dict[f'real_B'] = input[1] if len(input[1].shape) == 4 else input[1].unsqueeze(0)

            self.image_paths = ["./cache_physgen/" + f"building_{input[2]}.png" if self.opt.direction == 'AtoB' else f"{input[2]}_LAEQ.png"]
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if self.forward_passes < 10:
            print("New Input Images:")
            print(f"\n[Debug] Image (self.realA) stats:\n    - min: {self.real_A.min().item():.2f}\n    - max: {self.real_A.max().item():.2f}\n    - mean: {self.real_A.mean().item():.2f}\n    - shape: {self.real_A.shape}")
            print(f"\n[Debug] Image (self.real_B) stats:\n    - min: {self.real_B.min().item():.2f}\n    - max: {self.real_B.max().item():.2f}\n    - mean: {self.real_B.mean().item():.2f}\n    - shape: {self.real_B.shape}")
        else:
            pass    
            
        self.forward_passes += 1

    def set_current_epoch(self, epoch):
        new_epoch = self.current_epoch != epoch
        self.current_epoch = epoch

        if self.masked or self.post_masked:
            self.mask = shifted_mask(batch_size=self.batch_size, height=256, width=256, device=self.device, 
                                     region_size=(16, 16), epoch=self.current_epoch, shift_every_n_epochs=10,
                                     max_epoch=self.epochs)
            
        if self.activate_gan_mid_refinement:
            if epoch > int(self.epochs*0.2) and epoch < int(self.epochs*0.8):
                factor = 0.1 + 0.9 * (epoch - self.epochs * 0.2) / (self.epochs * 0.6)
                self.lambda_L1 = self.original_lambda_L1 * (1 - factor)  # slowly put down
                self.lambda_GAN = self.original_lambda_GAN * (1 + factor*4)  # GAN slightly increase
            else:
                self.lambda_GAN = self.original_lambda_GAN
                self.lambda_L1 = self.original_lambda_L1

        # update Loss Weighting
        # if new_epoch:
        #     self.lambda_GAN = min(epoch * 10.0, 200)
            # self.lambda_L1 += 0.5

            # if self.current_epoch/self.epochs >= 0.95:
            #     # self.masked = False 
            #     self.train_mask_area = False

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.opt.dataset_mode.lower() == "physgen":
            self.image_names_dict['fake_B'] = self.fake_B if len(self.fake_B.shape) == 4 else self.fake_B.unsqueeze(0)

    def forward_and_return(self):
        """Run forward pass and returns the output"""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.opt.dataset_mode.lower() == "physgen":
            self.image_names_dict['fake_B'] = self.fake_B if len(self.fake_B.shape) == 4 else self.fake_B.unsqueeze(0)
        return self.fake_B.detach()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if (not self.masked or self.current_epoch > self.epochs*0.8): # and (not self.post_masked or self.current_epoch > self.epochs*0.5): 
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_AB.detach())
            
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)

            if self.opt.wgangp:
                # WGAN loss
                self.loss_D_fake = pred_fake.mean()
                self.loss_D_real = -pred_real.mean()

                # Gradient penalty
                self.loss_D_gp = compute_gradient_penalty(
                                    self.netD, real_AB.detach(), fake_AB.detach(), device=self.device
                                )

                # Total loss
                self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp
            else:
                self.loss_D_fake = self.criterionGAN(pred_fake, False)
                self.loss_D_real = self.criterionGAN(pred_real, True)
                # combine loss and calculate gradients
                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            
            self.loss_D.backward()
        else:
            self.loss_D_real = 0.0
            self.loss_D_fake = 0.0
            self.loss_D_gp = 0.0
            self.loss_D = 0.0

        try:
            self.loss_D = self.loss_D.detach()
        except Exception:
            pass

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if (not self.masked or self.current_epoch > self.epochs*0.8):  # and (not self.post_masked or self.current_epoch > self.epochs*0.5): 
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)

            if self.opt.wgangp:
                self.loss_G_GAN = -pred_fake.mean()
            else:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0.0

        # calc second loss with masking and optional weighted loss
        if (self.masked and self.current_epoch <= self.epochs*0.8) or (self.post_masked and self.current_epoch > self.epochs*0.5):
            if self.weighted_loss:
                self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=self.mask)
            else:
                # self.loss_G_L1 = torch.mean(torch.abs(self.real_B - self.fake_B) * self.mask) 
                self.loss_G_L1 = masked_l1_loss(self.fake_B, self.real_B, self.mask)
        else:
            if self.weighted_loss:
                if self.calc_weight_map_for_weighted_loss:
                    mask = None
                else:
                    mask = torch.ones_like(self.real_B)
                self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=mask, data_idx=None, should_save=False, first_pass=self.first_loss_pass)
                self.first_loss_pass = False
            else:
                self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN * self.lambda_GAN + self.loss_G_L1 * self.lambda_L1
        self.loss_G.backward()
        self.loss_G = self.loss_G.detach()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def shrink_to_second_channel(self, input):
        if self.only_reflexions and input.shape[1] >= 2:
            input =  input[:, 1:, :, :]
        return input

def compute_gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=10.0):
    """
    Computes the gradient penalty used in WGAN-GP (Wasserstein GAN with Gradient Penalty).

    This function helps the discriminator (also called the critic) behave more smoothly and
    consistently. It does this by adding a penalty whenever the critic's output changes too
    sharply with small changes in the input — which is important for stable training.

    Here's what it does, step by step:

    1. It picks random points between real and fake data (a mix of both).
    2. It runs these mixed points through the discriminator.
    3. It measures how sensitive the discriminator is to these inputs by calculating gradients.
    4. If the gradients are too large or too small, it adds a penalty.
       Ideally, the gradient should have a length of 1.
    5. It returns this penalty as a loss term that can be added to the discriminator loss.

    Args:
        D (nn.Module): The discriminator (or critic) model.
        real_samples (Tensor): A batch of real data examples.
        fake_samples (Tensor): A batch of generated (fake) data.
        device (torch.device): The device (CPU or GPU) to run computations on.
        lambda_gp (float): A scaling factor for how strong the penalty should be.

    Returns:
        Tensor: A single scalar value representing the gradient penalty.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    # interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).detach()
    # interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)

    gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=fake,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty





