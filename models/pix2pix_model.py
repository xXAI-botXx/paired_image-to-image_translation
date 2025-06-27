import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import kornia

from .base_model import BaseModel
from . import networks

class WeightedCombinedLoss(nn.Module):
    def __init__(self, 
                 silog_lambda=0.5, 
                 weight_silog=0.5, 
                 weight_grad=10.0, 
                 weight_ssim=5.0,
                 weight_edge_aware=10.0,
                 weight_l1=1.0,
                 weight_var=1.0,
                 weight_range=1.0,
                 weight_blur=1.0):
        super().__init__()
        self.silog_lambda = silog_lambda
        self.weight_silog = weight_silog
        self.weight_grad = weight_grad
        self.weight_ssim = weight_ssim
        self.weight_edge_aware = weight_edge_aware
        self.weight_l1 = weight_l1
        self.weight_var = weight_var
        self.weight_range = weight_range
        self.weight_blur = weight_blur

        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.avg_loss_blur = 0
        self.steps = 0

        # Instantiate SSIMLoss module
        self.ssim_module = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        # self.ssim_module = kornia.losses.MS_SSIMLoss(reduction='mean')


    def silog_loss(self, pred, target, weight_map):
        eps = 1e-6
        pred = torch.clamp(pred, min=eps)
        target = torch.clamp(target, min=eps)
        
        diff_log = torch.log(target) - torch.log(pred)
        diff_log = diff_log * weight_map

        loss = torch.sqrt(torch.mean(diff_log ** 2) -
                          self.silog_lambda * torch.mean(diff_log) ** 2)
        return loss

    def gradient_l1_loss(self, pred, target, weight_map):
        # Create Channel Dimension
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        # Gradient in x-direction (horizontal -> dim=3)
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Gradient in y-direction (vertical -> dim=2)
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        loss_x = torch.mean(torch.abs(pred_grad_x - target_grad_x) * weight_x)
        loss_y = torch.mean(torch.abs(pred_grad_y - target_grad_y) * weight_y)
        
        # loss_x = F.l1_loss(pred_grad_x, target_grad_x) 
        # loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def ssim_loss(self, pred, target, weight_map):
        # SSIM returns similarity, so we subtract from 1
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # self.ssim_module = self.ssim_module.to(pred.device)
        return self.ssim_module(pred, target)

    def edge_aware_loss(self, pred, target, weight_map):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if weight_map.ndim == 3:
            weight_map = weight_map.unsqueeze(1)

        pred_grad_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_grad_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

        target_grad_x = torch.mean(torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:]), 1, keepdim=True)
        target_grad_y = torch.mean(torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :]), 1, keepdim=True)

        weight_x = weight_map[:, :, :, 1:] * weight_map[:, :, :, :-1]
        weight_y = weight_map[:, :, 1:, :] * weight_map[:, :, :-1, :]

        pred_grad_x *= torch.exp(-target_grad_x* weight_x) 
        pred_grad_y *= torch.exp(-target_grad_y* weight_y)

        # return (pred_grad_y.abs().mean() + target_grad_y.abs().mean())
        return (pred_grad_x.abs().mean() + pred_grad_y.abs().mean())

    def l1_loss(self, pred, target, weight_map):
        loss = torch.abs(target - pred) * weight_map
        return loss.mean()

    def variance_loss(self, pred, target):
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        return F.mse_loss(pred_var, target_var)
    
    def range_loss(self, pred, target):
        pred_min, pred_max = torch.min(pred), torch.max(pred)
        target_min, target_max = torch.min(target), torch.max(target)
        
        min_loss = F.mse_loss(pred_min, target_min)
        max_loss = F.mse_loss(pred_max, target_max)
        
        return min_loss + max_loss

    def blur_loss(self, pred, target):
        laplacian_kernel = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=pred.dtype, device=pred.device)

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        pred_lap = F.conv2d(pred, laplacian_kernel, padding=1)
        target_lap = F.conv2d(target, laplacian_kernel, padding=1)

        return F.l1_loss(pred_lap, target_lap)

    def forward(self, pred, target, weight_map=None):
        if type(weight_map) == type(None):
            weight_map = calc_weight_map(target)
        loss_silog = self.silog_loss(pred, target, weight_map)
        loss_grad = self.gradient_l1_loss(pred, target, weight_map)
        loss_ssim = self.ssim_loss(pred, target, weight_map)
        loss_l1 = self.l1_loss(pred, target, weight_map)
        loss_edge_aware = self.edge_aware_loss(pred, target, weight_map)
        loss_var = self.variance_loss(pred, target)
        loss_range = self.range_loss(pred, target)
        loss_blur = self.blur_loss(pred, target)

        self.avg_loss_silog += loss_silog
        self.avg_loss_grad += loss_grad
        self.avg_loss_ssim += loss_ssim
        self.avg_loss_l1 += loss_l1
        self.avg_loss_edge_aware += loss_edge_aware
        self.avg_loss_var += loss_var
        self.avg_loss_range += loss_range
        self.avg_loss_blur += loss_blur
        self.steps += 1

        total_loss = (
            self.weight_silog * loss_silog +
            self.weight_grad * loss_grad +
            self.weight_ssim * loss_ssim +
            self.weight_edge_aware * loss_edge_aware +
            self.weight_l1 * loss_l1 +
            self.weight_var * loss_var +
            self.weight_range * loss_range +
            self.weight_blur * loss_blur
        )
        return total_loss

    def step(self, epoch):
        self.avg_loss_silog = 0
        self.avg_loss_grad = 0
        self.avg_loss_ssim = 0
        self.avg_loss_l1 = 0
        self.avg_loss_edge_aware = 0
        self.avg_loss_var = 0
        self.avg_loss_range = 0
        self.avg_loss_blur = 0
        self.steps = 0

    def get_avg_losses(self):
        return (self.avg_loss_silog/self.steps,
                self.avg_loss_grad/self.steps,
                self.avg_loss_ssim/self.steps,
                self.avg_loss_l1/self.steps,
                self.avg_loss_edge_aware/self.steps,
                self.avg_loss_var/self.steps,
                self.avg_loss_range/self.steps,
                self.avg_loss_blur/self.steps
               )

    def get_dict(self, data_idx):
        loss_silog, loss_grad, loss_ssim, loss_l1, loss_edge_aware, loss_var, loss_range = self.get_avg_losses()
        return {
                f"{data_idx}_loss silog": loss_silog, 
                f"{data_idx}_loss grad": loss_grad, 
                f"{data_idx}_loss ssim": loss_ssim,
                f"{data_idx}_loss L1": loss_l1,
                f"{data_idx}_loss edge aware": loss_edge_aware,
                f"{data_idx}_loss var": loss_var,
                f"{data_idx}_loss range": loss_range,
                f"{data_idx}_loss blur": loss_blur,
                f"{data_idx}_weight loss silog": self.weight_silog, 
                f"{data_idx}_weight loss grad": self.weight_grad,
                f"{data_idx}_weight loss ssim": self.weight_ssim,
                f"{data_idx}_weight loss L1": self.weight_l1,
                f"{data_idx}_weight loss edge aware": self.weight_edge_aware,
                f"{data_idx}_weight loss var": self.weight_var,
                f"{data_idx}_weight loss range": self.weight_range,
                f"{data_idx}_weight loss blur": self.weight_blur
               }

def calc_weight_map(target):
    values, counts = torch.unique(target.flatten(), return_counts=True)
    all_counts = counts.sum().float()
    
    # weight_factor = 2.0
    # weights = {values[idx].item(): max(torch.exp( ( (1-(counts[idx].item()/all_counts))) *weight_factor), 0.0001) for idx in range(len(values))}
    
    weights = {values[idx].item(): 255.0/counts[idx].item() for idx in range(len(values))}

    # print(f"Weights:")
    # for cur_value, cur_counts in list(sorted(weights.items(), key=lambda x:x[0])):
    #     print('    - '+str(round(cur_value, 4))+': '+str(cur_counts.item()))

    weights_map = torch.zeros_like(target, dtype=torch.float)
    for cur_value in values:
        cur_value = cur_value.item()
        weights_map[target == cur_value] = weights[cur_value]

    return weights_map

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
            parser.add_argument('--wgangp', action='store_true', help='Should use WGAN-GP')
            parser.add_argument('--masked', action='store_true', help='Should mask with the target and threshold at 0')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.masked = opt.masked
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
        
        self.lambda_GAN = 1.0
        self.epochs_with_gan = 0
        self.forward_passes = 0
        self.current_epoch = 0

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

        # pix2pix_1_0_residual_few_bui_masked_weight_loss_4
        self.weighted_loss = WeightedCombinedLoss( 
                                            weight_silog=10.0, 
                                            weight_grad=100.0, 
                                            weight_ssim=0.0,
                                            weight_edge_aware=100.0,
                                            weight_l1=1000.0,
                                            weight_var=10.0,
                                            weight_range=10.0,
                                            weight_blur=0.0
                                )

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
        return self.fake_B

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
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

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        if self.opt.wgangp:
            self.loss_G_GAN = -pred_fake.mean()
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # Second, G(A) = B
        # L1 loss masked
        if self.masked:
            # Compute pixel-wise absolute difference
            l1_diff = torch.abs(self.fake_B - self.real_B)

            # Create mask where real_B > 0
            mask = (self.real_B > 0).float() # (self.real_B > 0).astype(np.uint8) * 255

            # Apply mask and normalize
            if False:
                masked_l1 = l1_diff * mask
                num_masked = torch.clamp(mask.sum(), min=1.0)  # prevent division by zero
                self.loss_G_L1 = masked_l1.sum() / num_masked
            else:
                if self.train_mask_area:
                    self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=mask) 
                    # self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=None) 
                else:
                    inverted_mask = 1 - mask
                    self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=inverted_mask) 
        else:
            # No masking
            # self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=None) 
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN * self.lambda_GAN + self.loss_G_L1 * self.opt.lambda_L1
        self.loss_G.backward()

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





