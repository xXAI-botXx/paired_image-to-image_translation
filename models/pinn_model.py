import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from timm.models.vision_transformer import vit_base_patch16_224
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

import kornia

from .base_model import BaseModel
from . import networks

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = conv_block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = conv_block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bottleneck = self.bottleneck(p4)

        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv_up2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv_up3(u3)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv_up4(u4)

        return self.out_conv(u4)

class PINN(nn.Module):
    def __init__(self, in_channels, out_channels, k_value=None):
        super(PINN, self).__init__()
        self.unet = UNet(in_channels, out_channels)
        self.k_value = k_value # Wave number for Helmholtz equation

    def forward(self, x):
        return self.unet(x)

    def helmholtz_loss(self, u_pred, x_coords, y_coords, sample_points=None):
        # Ensure u_pred requires gradients for autograd
        u_pred_grad = u_pred.clone()
        u_pred_grad.requires_grad_(True)

        # Ensure coordinates require gradients
        x_coords.requires_grad_(True)
        y_coords.requires_grad_(True)

        if sample_points is not None:
            # Randomly sample points for physics loss calculation
            # Flatten the spatial dimensions to easily sample points
            u_pred_flat_spatial = u_pred_grad.view(u_pred_grad.shape[0], u_pred_grad.shape[1], -1)
            x_coords_flat_spatial = x_coords.view(x_coords.shape[0], x_coords.shape[1], -1)
            y_coords_flat_spatial = y_coords.view(y_coords.shape[0], y_coords.shape[1], -1)

            # Generate random indices for sampling
            num_pixels = u_pred_flat_spatial.shape[2]
            indices = torch.randperm(num_pixels, device=u_pred_grad.device)[:sample_points]

            # Select sampled points for each batch and channel
            u_pred_sampled = u_pred_flat_spatial[:, :, indices]
            x_coords_sampled = x_coords_flat_spatial[:, :, indices]
            y_coords_sampled = y_coords_flat_spatial[:, :, indices]

            # Flatten for autograd.grad, ensuring each element is treated independently
            u_pred_flat = u_pred_sampled.view(-1)
            x_flat = x_coords_sampled.view(-1)
            y_flat = y_coords_sampled.view(-1)
        else:
            u_pred_flat = u_pred_grad.reshape(-1)
            x_flat = x_coords.view(-1)
            y_flat = y_coords.view(-1)

        # Calculate first derivatives
        # Using torch.ones_like(u_pred_flat) for grad_outputs ensures that the sum of gradients is computed.
        # create_graph=True is essential for computing second derivatives.
        print("CHECK", u_pred_flat.requires_grad, x_flat.requires_grad)
        du_dx = autograd.grad(u_pred_flat, x_flat, grad_outputs=torch.ones_like(u_pred_flat), create_graph=True, allow_unused=True)[0]
        print("du_dx.requires_grad:", du_dx.requires_grad)
        print("du_dx.grad_fn:", du_dx.grad_fn)
        print("du_dx is None:", du_dx is None)
        print("du_dx.shape:", None if du_dx is None else du_dx.shape)
        du_dy = autograd.grad(u_pred_flat, y_flat, grad_outputs=torch.ones_like(u_pred_flat), create_graph=True, allow_unused=True)[0]

        # Handle None gradients by replacing them with zeros
        du_dx = du_dx if du_dx is not None else torch.zeros_like(u_pred_flat)
        du_dy = du_dy if du_dy is not None else torch.zeros_like(u_pred_flat)

        # Calculate second derivatives (Laplacian)
        # Again, create_graph=True is needed for the subsequent backward pass through the physics loss.
        d2u_dx2 = autograd.grad(du_dx, x_flat, grad_outputs=torch.ones_like(du_dx), create_graph=True, allow_unused=True)[0]
        d2u_dy2 = autograd.grad(du_dy, y_flat, grad_outputs=torch.ones_like(du_dy), create_graph=True, allow_unused=True)[0]

        # Handle None gradients for second derivatives
        d2u_dx2 = d2u_dx2 if d2u_dx2 is not None else torch.zeros_like(u_pred_flat)
        d2u_dy2 = d2u_dy2 if d2u_dy2 is not None else torch.zeros_like(u_pred_flat)

        laplacian_u = d2u_dx2 + d2u_dy2

        if self.k_value is None:
            return torch.zeros_like(u_pred_flat).mean()
        else:
            # If sampling, u_pred_grad needs to be sampled too for the k_value term
            if sample_points is not None:
                physics_loss = laplacian_u + (self.k_value**2) * u_pred_sampled.reshape(-1)
            else:
                physics_loss = laplacian_u + (self.k_value**2) * u_pred_flat
            return torch.mean(physics_loss**2)

    def boundary_loss(self, u_pred, osm_image, target_value=0.0):
        boundary_mask = (osm_image == 0).float()
        return torch.mean((u_pred * boundary_mask - target_value * boundary_mask)**2)

def mape_loss(prediction, target, eps=1e-6):
    not_null_target = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((prediction - target) / not_null_target))

def calc_pinn_loss(model, input_image, target):
    B, C, H, W = input_image.shape
    x = torch.linspace(0, 1, W, device=input_image.device, requires_grad=True).reshape(1, 1, 1, W).expand(B, -1, H, -1)
    y = torch.linspace(0, 1, H, device=input_image.device, requires_grad=True).reshape(1, 1, H, 1).expand(B, -1, -1, W)

    x.requires_grad_(True)
    y.requires_grad_(True)

    input_with_coords = torch.cat([input_image, x, y], dim=1)  # Shape: [B, C+2, H, W]
    input_with_coords.requires_grad_(True)

    prediction = model(input_with_coords)
    prediction.requires_grad_(True)

    cur_helmholtz_loss = model.helmholtz_loss(prediction, x, y) #, sample_points=1000)
    # print(f"Helmholtz Loss (sampled): {cur_helmholtz_loss.item()}")

    cur_boundary_loss = model.boundary_loss(prediction, input_image)
    # print(f"Boundary Loss: {cur_boundary_loss.item()}")

    cur_mape_loss = mape_loss(prediction, target)
    # print(f"MAPE Loss: {cur_mape_loss.item()}")

    lambda_physics = 0.1
    lambda_boundary = 0.5
    lambda_mape = 1.0
    total_loss = cur_helmholtz_loss * lambda_physics + \
                 cur_boundary_loss * lambda_boundary + \
                 cur_mape_loss * lambda_mape
    # print(f"Total Loss (example): {total_loss.item()}")

    return total_loss

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



class PINNModel(BaseModel):
    """ This class implements the stacked ResUNet model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--wgangp', action='store_true', help='Should use WGAN-GP')
            parser.add_argument('--masked', action='store_true', help='Should mask with the target and threshold at 0')

        return parser

    def __init__(self, opt):
        """Initialization.

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = PINN(in_channels=3, out_channels=1, k_value=9.16).to(device)


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

        if self.forward_passes == 0:
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
        input_image = self.real_A
        B, C, H, W = input_image.shape
        self.x = torch.linspace(0, 1, W, device=input_image.device, requires_grad=True).reshape(1, 1, 1, W).expand(B, -1, H, -1)
        self.y = torch.linspace(0, 1, H, device=input_image.device, requires_grad=True).reshape(1, 1, H, 1).expand(B, -1, -1, W)

        self.x.requires_grad_(True)
        self.y.requires_grad_(True)

        coords = torch.cat([self.x, self.y], dim=1)  # Shape: [B, 2, H, W]
        coords.requires_grad_(True)
        input_with_coords = torch.cat([input_image, coords.to(input_image.device)], dim=1)  # Shape: [B, C+2, H, W]
        input_with_coords.requires_grad_(True)

        self.fake_B = self.generator(input_with_coords)
        # self.fake_B = self.generator(self.real_A)  # G(A)
        if self.opt.dataset_mode.lower() == "physgen":
            self.image_names_dict['fake_B'] = self.fake_B if len(self.fake_B.shape) == 4 else self.fake_B.unsqueeze(0)
        return self.fake_B

    def forward_and_return(self):
        """Run forward pass and returns the output"""
        return self.forward()

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
        # No masking
        # self.loss_G_L1 = self.weighted_loss(pred=self.fake_B, target=self.real_B, weight_map=None) 
        # self.loss_G_L1 = calc_pinn_loss(model=self.generator, input_image=self.real_A, prediction=self.fake_B, target=self.real_B, x=self.x, y=self.y)
        self.loss_G_L1 = calc_pinn_loss(model=self.generator, input_image=self.real_A, target=self.real_B)
        
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





