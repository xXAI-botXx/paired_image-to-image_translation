"""
The pix2phys model builds upon the concept of the pix2pix
but improves the architecture to work for 
Physics-Based Supervised One-to-One Image Translation for predicting sound propagation.

Architecture:
        Input: 256x256xC (e.g., geometry, material, boundary data)
           ↓
  CNN Encoder (U-Net / ResNet-style): extracts local features
           ↓
      FNO Layer(s): models long-range frequency-aware field behavior
           ↓
  Latent Transformer Block:
                    - Attention over spatial regions
                    - Acts as a saliency filter: learns where to focus
           ↓
     SIREN Decoder (predicts continuous field from coordinates + latent features)
           ↓
        Output: 256x256x1 (e.g., sound pressure / SPL map)


Explaination:
- CNN-Encoder: Captures local features
- FNO-Encoder: Models global frequency interactions
- Latent Space Transformer: Learns saliency dynamically in latent space
- SIREN-Decoder: Refines sharp, continuous signal edges
"""

# ---------------
# --- Imports ---
# ---------------
from .base_model import BaseModel
from . import networks

# pip install einops
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchvision.models as models
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, resnet18



# -----------------------
# --- Generator Model ---
# -----------------------

# CNN Encoder

class ConvNeXtEncoder(nn.Module):
    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_base(weights=weights)

        # Change from 3 channels to 1
        orig_conv = backbone.features[0][0]  # Conv2d(3, 96, kernel_size=4, stride=4)
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )

        #     Initialize new weights (e.g., average over RGB if pretrained)
        if pretrained and in_channels == 1:
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
            if orig_conv.bias is not None:
                new_conv.bias.data = orig_conv.bias.data.clone()

        #     Replace the original conv layer
        backbone.features[0][0] = new_conv

        # Set used Layers
        self.stem = backbone.features[0]
        self.stage1 = backbone.features[1]
        self.stage2 = backbone.features[2]
        self.stage3 = backbone.features[3]  # Output: [B, 576, 16, 16]
        self.stage4 = backbone.features[4]  # Would downsample to 8x8
        self.stage5 = backbone.features[5]

        # print(f"ConvNext features: {backbone.features}")

        self.proj = nn.Conv2d(512, 512, 1)  # Project to match latent_dim

    def forward(self, x):
        skips = []

        x = self.stem(x)    # [B, 128, 128, 128]
        skips += [x]
        # print(f"Stem Shape x: {x.shape}")

        x = self.stage1(x)  # [B, 128, 128, 128]
        skips += [x]
        # print(f"Stage 1 - Shape x: {x.shape}")

        x = self.stage2(x)  # [B, 256, 64, 64]
        skips += [x]
        # print(f"Stage 2 - Shape x: {x.shape}")

        x = self.stage3(x)  # [B, 256, 64, 64]
        # print(f"Stage 3 - Shape x: {x.shape}")

        x = self.stage4(x)  # [B, 512, 32, 32]
        # print(f"Stage 4 - Shape x: {x.shape}")

        x = self.stage5(x)  # [B, 512, 32, 32]
        # print(f"Stage 5 - Shape x: {x.shape}")

        x = self.proj(x)    # [B, 512, 16, 16]
        # print(f"Projection Shape x: {x.shape}")
        return x, skips

def process_skips(skip_feats, target_size=(256, 256)):
    processed = []
    for feat in skip_feats:
        upsampled = F.interpolate(feat, size=target_size, mode='bilinear')  # [B, C, H, W] → [B, C, 256, 256]
        flat = rearrange(upsampled, 'b c h w -> b (h w) c')  # [B, N, C]
        processed += [flat]
    return torch.cat(processed, dim=-1)  # [B, N, C_total]

# class ResNetEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         backbone = models.resnet18(pretrained=True)
#         self.encoder = nn.Sequential(
#             backbone.conv1,  # [B, 64, 128, 128]
#             backbone.bn1,
#             backbone.relu,
#             backbone.maxpool,  # [B, 64, 64, 64]
#             backbone.layer1,   # [B, 64, 64, 64]
#             backbone.layer2,   # [B, 128, 32, 32]
#             backbone.layer3,   # [B, 256, 16, 16]
#         )

#     def forward(self, x):
#         return self.encoder(x)  # [B, 256, 16, 16]

# class SimpleCNNEncoder(nn.Module):
#     def __init__(self, in_channels=3, base_channels=64):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 128x128
#             nn.ReLU(),
#             nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 64x64
#             nn.ReLU(),
#             nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 32x32
#             nn.ReLU(),
#             nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # 16x16
#             nn.ReLU(),
#         )

    def forward(self, x):
        return self.encoder(x)  # [B, C, 16, 16]


# Helper module for FNO
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')  # [B, C, H, W//2 + 1]

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')  # back to real space
        return x

# Fourier Neural Operator Network
class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, width=512, modes1=12, modes2=12, depth=4):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, width, 1)

        self.spectral_layers = nn.ModuleList()
        self.pointwise_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.pointwise_layers.append(nn.Conv2d(width, width, 1))

        # Keep output shape and dimension same
        self.out_proj = nn.Conv2d(width, out_channels, 1) if out_channels != width else nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        x = self.in_proj(x)
        for spec, point in zip(self.spectral_layers, self.pointwise_layers):
            x = self.act(spec(x) + point(x))
        x = self.out_proj(x)
        return x



# Transformer for latent space -> indirect saliency map
class LatentTransformer(nn.Module):
    def __init__(self, dim, heads=4, depth=2):
        super().__init__()
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads),
            num_layers=depth
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.layers(x)
        return x  # [B, HW, C]



# Helper for SIREN Decoder
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-
                    (6 / self.linear.in_features) ** 0.5 / self.omega_0,
                    (6 / self.linear.in_features) ** 0.5 / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# SIREN Decoder
class SIRENDecoder(nn.Module):
    def __init__(self, coord_dim, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(coord_dim + latent_dim, hidden, is_first=True),
            SineLayer(hidden, hidden),
            SineLayer(hidden, hidden),
            nn.Linear(hidden, 1)  # output scalar value per coordinate
        )

    def forward(self, coords, latent_features):
        # coords: [B, N, 2], latent_features: [B, N, C]
        if len(coords.shape) == 2:
            coords = torch.unsqueeze(coords, dim=0)
        x = torch.cat([coords, latent_features], dim=-1)
        return self.net(x)  # [B, N, 1]



# Whole Generator Module 
class HexaWaveNetGenerator(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = ConvNeXtEncoder(pretrained=True)    # SimpleCNNEncoder(in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        self.transformer = LatentTransformer(dim=512)
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=512*2)

    def forward(self, image):
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # if image.shape[1] == self.in_channels:  # [B, C, H, W] -> [B, H, W, C]
        #     image = image.permute(0, 2, 3, 1)
        

        batch_size = image.shape[0]

        # print(f"[Info] Model got Image with shape: {image.shape}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        # print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        # print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        # print(f"feat_comb Shape: {feat_comb.shape}")

        # >>> Latent Space <<<
        #     ------------
        # Transformer latent attention
        latent_seq = self.transformer(feat_comb)  # [B, 1024, 512]
        B, HW, C = latent_seq.shape
        # print(f"Latent Transformer Shape: {latent_seq.shape}")
        assert HW == 16*16, f"Expected 256 tokens, got {HW}"
        assert C == 512, f"Expected 512 channels, got {C}"

        # >>> Upsample Latent Space <<<
        #     ---------------------
        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        # print(f"Coordinate Shapes: {self.coords.shape}")        
        # H_W, N = self.coords.shape
        latent_map = latent_seq.reshape(B, 16, 16, C).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        # print(f"Latent Map Shape: {latent_map.shape}")

        # Upscaling Latent Space
        latent_up = F.interpolate(latent_map, size=(self.image_size, self.image_size), mode='bilinear')  # [B, C, 256, 256]
        
        # Latent Space in coordinate format for SIREN Decoder
        latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]
        # print(f"Latent Flat Shape: {latent_flat.shape}")

        # >>> Skip Connection <<<
        #     ---------------
        # Process skip features
        skip_flat = process_skips(skips, target_size=(self.image_size, self.image_size))  # [B, N, C_skips]
        # print(f"Skip Flat Shape: {skip_flat.shape}")

        # Combine latent + skips
        full_features = torch.cat([latent_flat, skip_flat], dim=-1)  # [B, N, C_total]
        # print(f"Full Features Shape: {full_features.shape}")

        # # Reduce features
        # self.feature_proj = nn.Linear(combined_dim, latent_dim)  # optional
        # reduced_features = self.feature_proj(full_features)  # [B, N, latent_dim]

        # >>> Decoding <<<
        #     --------
        # using SIREN
        output = self.decoder(self.coords, full_features)  # [B, N, 1]
        # print(f"Decoder Shape: {output.shape}")

        # >>> Reshaping <<<
        #     ---------
        output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)
        return output

def get_normalized_grid(H=256, W=256, device="cuda"):
    # Creates a normalized meshgrid from -1 to 1
    y = torch.linspace(-1, 1, H).to(device)
    x = torch.linspace(-1, 1, W).to(device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    coords = coords.view(-1, 2)  # [H*W, 2]
    return coords


# -----------------
# --- GAN Model ---
# -----------------
class HexaWaveNetModel(BaseModel):
    """ This class implements the pix2phys model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    based on pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf

    Additions:
    - Wassterstein GAN - Gradient Penalty as loss
    - Generator:
        - CNN Encoder + FNO Encoder
        - Transformer in latent space
        - SIREN Decoder 
        - Skip Connections?
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2phys, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--gan_disable_epoch', type=int, default=50, help='Epoch after which GAN loss is disabled')

        return parser

    def __init__(self, opt):
        """Initialize the pix2phys class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
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
        self.netG = networks.init_net(HexaWaveNetGenerator(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size), 
                                      opt.init_type, 
                                      opt.init_gain, 
                                      self.gpu_ids)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)    # not in use here
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.current_epoch = 0

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
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def forward_and_return(self):
        """Run forward pass and returns the output"""
        self.fake_B = self.netG(self.real_A)  # G(A)
        return self.fake_B

    # Maybe change to calculate this over patches?
    def backward_D(self):
        """
        Calculate GAN loss for the discriminator.

        Wasserstein loss with gradient penalty.
        """
        # print(f"self.real_A: {self.real_A.shape}")
        # print(f"self.fake_B: {self.fake_B.shape}")
        # print(f"self.real_B: {self.real_B.shape}")

        fake_AB = torch.cat((self.real_A, self.fake_B), dim=1)
        real_AB = torch.cat((self.real_A, self.real_B), dim=1)

        # print(f"fake_AB: {fake_AB.shape}")
        # print(f"self.real_A: {self.real_A.shape}")
        # print(f"self.real_B: {self.fake_B.shape}")

        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)

        # WGAN loss
        self.loss_D_real = -pred_real.mean()
        self.loss_D_fake = pred_fake.mean()

        # Gradient penalty
        self.loss_D_gp = compute_gradient_penalty(
            self.netD, real_AB.data, fake_AB.data, device=self.device
        )

        # Total loss
        self.loss_D = self.loss_D_real + self.loss_D_fake + self.loss_D_gp
        self.loss_D.backward()

    def backward_G(self):
        """
        Generator loss with optional GAN loss
        """
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        if self.current_epoch < self.opt.gan_disable_epoch:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = -pred_fake.mean()
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G_GAN = torch.tensor(0.0, device=self.device)  # for logging
            # Edge-aware Gradient Loss
            lambda_grad = 30.0 # getattr(self.opt, "lambda_grad", 10.0)  # set default if not specified
            self.loss_G_grad = self.compute_gradient_loss(self.fake_B, self.real_B) * lambda_grad
            self.loss_G = self.loss_G_L1 + self.loss_G_grad

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Discrimantor
        if self.current_epoch < self.opt.gan_disable_epoch:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # Generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_gradient_loss(self, fake, real):
        def gradient(img):
            dx = img[:, :, :, 1:] - img[:, :, :, :-1]
            dy = img[:, :, 1:, :] - img[:, :, :-1, :]
            return dx, dy

        dx_fake, dy_fake = gradient(fake)
        dx_real, dy_real = gradient(real)
        loss = torch.mean(torch.abs(dx_fake - dx_real)) + torch.mean(torch.abs(dy_fake - dy_real))
        return loss

# Helper for loss
def compute_gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=10.0):
    """
    Calculates the gradient penalty loss for WGAN GP
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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

    





