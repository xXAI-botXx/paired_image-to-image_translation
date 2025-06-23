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

import math
import numpy

# pip install einops
from einops import rearrange

# pip install pytorch-msssim
import pytorch_msssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn import Transformer
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

def process_skips(skip_feats, target_size=(256, 256), flatten=True):
    processed = []
    for feat in skip_feats:
        upsampled = F.interpolate(feat, size=target_size, mode='bilinear')  # [B, C, H, W] → [B, C, 256, 256]
        if flatten:
            upsampled = rearrange(upsampled, 'b c h w -> b (h w) c')  # [B, N, C]
        processed += [upsampled]

    if flatten:
        return torch.cat(processed, dim=-1)  # [B, N, C_total]
    else:
        return torch.cat(processed, dim=1)  # [B, C_total, 256, 256]

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

class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # 16x16
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)  # [B, C, 16, 16]

class SimpleCNNEncoderWithSkips(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels, base_channels, 4, 2, 1)  # 128x128
        self.act1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)  # 64x64
        self.act2 = nn.ReLU()
            
        self.cnn3 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)  # 32x32
        self.act3 = nn.ReLU()
            
        self.cnn4 = nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)  # 16x16
        self.act4 = nn.ReLU()

    def forward(self, x):
        skips = []

        x = self.cnn1(x)
        x = self.act1(x)
        skips += [x]  # [B, base_channels, 128, 128] 

        x = self.cnn2(x)
        x = self.act2(x)
        skips += [x]  # [B, base_channels*2, 64, 64]

        x = self.cnn3(x)
        x = self.act3(x)
        skips += [x]  # [B, base_channels*4, 32, 32]

        x = self.cnn4(x)
        x = self.act4(x)
        output = x  # [B, base_channels*8, 16, 16] -> for example: [B, 512, 16, 16]

        return output, skips


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

# class LatentToImageTransformerDecoder(nn.Module):
#     def __init__(self, latent_dim, image_channels=3, hidden_dim=512, num_heads=8, num_layers=6, dropout=0.1, image_size=256):
#         super().__init__()

#         self.image_size = image_size
#         self.num_queries = image_size * image_size

#         # Learnable query tokens (each corresponds to a pixel)
#         self.query_embed = nn.Parameter(torch.randn(1, self.num_queries, hidden_dim))

#         self.latent_proj = nn.Linear(latent_dim, hidden_dim)

#         decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#         # Final projection to RGB or other channels
#         self.output_proj = nn.Linear(hidden_dim, image_channels)

#     def forward(self, memory):
#         """
#         memory: [B, S, latent_dim] -> typically S=256 for 16x16
#         Returns: [B, C, H, W]
#         """
#         B = memory.size(0)
#         memory = self.latent_proj(memory).permute(1, 0, 2)  # [S, B, H]
        
#         # Expand queries for batch
#         queries = self.query_embed.expand(B, -1, -1).permute(1, 0, 2)  # [N, B, H]

#         decoded = self.decoder(queries, memory)  # [N, B, H]
#         decoded = decoded.permute(1, 0, 2)  # [B, N, H]

#         # Project to image channels
#         out = self.output_proj(decoded)  # [B, N, C]
#         out = out.permute(0, 2, 1).reshape(B, -1, self.image_size, self.image_size)  # [B, C, H, W]
#         return out


# # Another Transformer Decoder
# class LatentToImageTransformerDecoder(nn.Module):
#     def __init__(self, latent_dim, image_channels=3, hidden_dim=256, num_heads=8, num_layers=6, dropout=0.1):
#         super(LatentToImageTransformerDecoder, self).__init__()

#         # Latent representation projection to transformer input dimension
#         self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        
#         # Positional Encoding
#         self.positional_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # Learnable positional encoding
#         # self.positional_encoding = nn.Parameter(torch.randn(1, H*W, hidden_dim))

#         # Transformer decoder layers
#         self.transformer_decoder = Transformer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             num_encoder_layers=0,  # No encoder, only decoder
#             num_decoder_layers=num_layers,
#             dim_feedforward=hidden_dim * 4,
#             dropout=dropout
#         )

#         # Final convolution to generate image
#         self.conv_out = nn.Conv2d(hidden_dim, image_channels, kernel_size=3, stride=1, padding=1)

#     def forward(self, latent_vector):
#         # Project latent vector to the decoder's input space
#         latent_vector = self.latent_projection(latent_vector).unsqueeze(0)  # Add batch dimension

#         # Create a sequence of 'tokens' with the same latent dimensions
#         sequence_len = 16  # A fixed length of sequence for latent tokens (this can vary)
#         tokens = latent_vector.repeat(1, sequence_len, 1)

#         # Add positional encoding
#         tokens += self.positional_encoding
        
#         # Pass through the transformer decoder
#         decoded_tokens = self.transformer_decoder(
#             tgt=tokens,  # Latent tokens as the target
#             memory=None   # No encoder, so no memory
#         )
        
#         # Reshape the output to be of the shape (B, C, H, W) for image generation
#         decoded_tokens = decoded_tokens.squeeze(0)  # Remove batch dimension
        
#         # Reshape tokens to a (B, C, H, W) format
#         img = self.conv_out(decoded_tokens.permute(0, 2, 1).view(1, -1, 4, 4))  # Final reshaping to image space
        
#         return img

# CNN Decoder
class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(CNNDecoder, self).__init__()

        self.up1 = self._up_block(in_channels, 256)
        self.up2 = self._up_block(256, 128)
        self.up3 = self._up_block(128, 64)
        self.up4 = self._up_block(64, out_channels)  # Final output size: 256x256

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)  # 16x16 -> 32x32
        x = self.up2(x)  # 32x32 -> 64x64
        x = self.up3(x)  # 64x64 -> 128x128
        x = self.up4(x)  # 128x128 -> 256x256
        return x  # [B, out_channels, 256, 256]

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
class HexaWaveNetGenerator_1(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = ConvNeXtEncoder(pretrained=True)    # SimpleCNNEncoder(in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        self.transformer = LatentTransformer(dim=512)

        # self.transformer_decoder = LatentTransformerDecoder(dim=512)
        # self.feature_proj = nn.Linear(1024, 512)
        self.cnn_decoder = CNNDecoder(in_channels=512, out_channels=64)
        # self.latent_to_image_decoder = LatentToImageTransformerDecoder(
        #                                     latent_dim=512+64,  # This should be latent_flat + skip_flat channels
        #                                     image_channels=self.in_channels,
        #                                     hidden_dim=512,
        #                                     num_heads=8,
        #                                     num_layers=6
        #                                 )
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=512*2)
        self.output_activation = nn.Tanh()

    def forward(self, image, should_print=False):
        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # if image.shape[1] == self.in_channels:  # [B, C, H, W] -> [B, H, W, C]
        #     image = image.permute(0, 2, 3, 1)
        

        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net")
            print(f"[Debug] Image stats - min: {image.min().item():.2f}, max: {image.max().item():.2f}, mean: {image.mean().item():.2f}")
            print(f"[Info] Model got Image with shape: {image.shape}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        if should_print:
            print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        if should_print:
            print(f"feat_comb Shape: {feat_comb.shape}")

        # >>> Latent Space <<<
        #     ------------
        # Transformer latent attention
        latent_seq = self.transformer(feat_comb)  # [B, 1024, 512]
        B, HW, C = latent_seq.shape
        if should_print:
            print(f"Latent Transformer Shape: {latent_seq.shape}")
        assert HW == 16*16, f"Expected 256 tokens, got {HW}"
        assert C == 512, f"Expected 512 channels, got {C}"

        # >>> Upsample Latent Space <<<
        #     ---------------------
        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        if should_print:
            print(f"Coordinate Shapes: {self.coords.shape}")        

        latent_map = latent_seq.reshape(B, 16, 16, C).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        if should_print:
            print(f"Latent Map Shape: {latent_map.shape}")

        # Upscaling Latent Space
        latent_up = F.interpolate(latent_map, size=(self.image_size, self.image_size), mode='bilinear')  # [B, C, 256, 256]

        # Latent Space in coordinate format for SIREN Decoder
        latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]
        if should_print:
            print(f"Latent Flat Shape: {latent_flat.shape}")

        # >>> Skip Connection <<<
        #     ---------------
        # Process skip features
        skip_flat = process_skips(skips, target_size=(self.image_size, self.image_size))  # [B, N, C_skips]
        if should_print:
            print(f"Skip Flat Shape: {skip_flat.shape}")

        # Combine latent + skips
        full_features = torch.cat([latent_flat, skip_flat], dim=-1)  # [B, N, C_total]
        if should_print:
            print(f"Full Features Shape: {full_features.shape}")

        # # Reduce features
        # self.feature_proj = nn.Linear(combined_dim, latent_dim)  # optional
        # reduced_features = self.feature_proj(full_features)  # [B, N, latent_dim]

        # >>> Decoding <<<
        #     --------
        # # using Transformer
        # output = self.latent_to_image_decoder(full_features)  # returns [B, C, H, W]
        decoding_cnn = self.cnn_decoder(latent_map)
        if should_print:
            print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # Decode with SIREN
        decoding_siren = self.decoder(self.coords, full_features)
        # Scalars (single values) per pixel to image array -> N = HxW
        decoding_siren = decoding_siren.view(batch_size, self.in_channels, self.image_size, self.image_size)  # [B, N, C] -> [B, 1, H, W]
        if should_print:
            print(f"SIREN Decoder Shape: {decoding_siren.shape}")

        # MLP take both decodings + the input image and generate the image
        combined = torch.cat([decoding_siren, decoding_cnn], dim=1)  # [B, 2C, H, W]
        fusion = nn.Conv2d(self.in_channels+64, self.in_channels, kernel_size=3, padding=1).to(self.device)
        output = fusion(combined)
        output = self.output_activation(output)

        # >>> Reshaping <<<
        #     ---------
        output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        if should_print:
            print(f"Final Output Shape: {output.shape}")
            print(f"\n[Debug] Output Image stats - min: {output.min().item():.2f}, max: {output.max().item():.2f}, mean: {output.mean().item():.2f}")
            print(f"Finished Forwarding Hexa Wave Net\n{'-'*32}\n")
            
        return output

class HexaWaveNetGenerator_2(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = ConvNeXtEncoder(pretrained=True)    # SimpleCNNEncoder(in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        self.transformer = LatentTransformer(dim=512)

        self.cnn_decoder = CNNDecoder(in_channels=512*2, out_channels=1)
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=1)
        self.output_activation = nn.Tanh()

    def forward(self, image, should_print=False):
        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net")
            print(f"[Debug] Image stats - min: {image.min().item():.2f}, max: {image.max().item():.2f}, mean: {image.mean().item():.2f}")
            print(f"[Info] Model got Image with shape: {image.shape}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        if should_print:
            print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = torch.cat([feat_cnn, feat_fno], dim=1)
        # feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        if should_print:
            print(f"feat_comb Shape: {feat_comb.shape}")


        # >>> Decoding <<<
        #     --------
        decoding_cnn = self.cnn_decoder(feat_comb)
        if should_print:
            print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # Decode with SIREN
        # feat_comb_flat = feat_comb.view(batch_size, -1, 512*2)  # Flatten to [B, H*W, 512]
        # [B, C, H, W] -> [B, N, C]
        image_flat = image.view(batch_size, -1, self.in_channels)
        decoding_siren = self.decoder(self.coords, image_flat)
        # Scalars (single values) per pixel to image array -> N = HxW
        decoding_siren = decoding_siren.view(batch_size, self.in_channels, self.image_size, self.image_size)  # [B, N, C] -> [B, 1, H, W]
        if should_print:
            print(f"SIREN Decoder Shape: {decoding_siren.shape}")

        # MLP/CNN take both decodings + the input image and generate the image
        combined = torch.cat([decoding_siren, decoding_cnn, image], dim=1)  # [B, 2C, H, W]
        fusion = nn.Conv2d(self.in_channels*3, self.in_channels, kernel_size=3, padding=1).to(self.device)
        output = self.output_activation(fusion(combined))

        # >>> Reshaping <<<
        #     ---------
        output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        if should_print:
            print(f"Final Output Shape: {output.shape}")
            print(f"\n[Debug] Output Image stats - min: {output.min().item():.2f}, max: {output.max().item():.2f}, mean: {output.mean().item():.2f}")
            print(f"Finished Forwarding Hexa Wave Net\n{'-'*32}\n")
            
        return output

class HexaWaveNetGenerator_3(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.forward_passes = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = SimpleCNNEncoder(self.in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        # self.transformer = LatentTransformer(dim=512)

        self.cnn_decoder = CNNDecoder(in_channels=512*2, out_channels=1)
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=1)

        self.fusion_cnn = nn.Conv2d(self.in_channels*3, 64, kernel_size=3, padding=1).to(self.device)

        self.mlp_head = nn.Sequential(
                                nn.Linear(in_features=64, out_features=32, device=self.device, bias=True),
                                nn.Linear(in_features=32, out_features=16, device=self.device, bias=True),
                                nn.Linear(in_features=16, out_features=self.in_channels, device=self.device, bias=True)
                        )
        self.output_activation = nn.Tanh()

    def forward(self, image):
        if self.forward_passes == 0:    # or numpy.random.rand() > 0.9
            should_print = True
        else:
            should_print = False

        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net")
            print(f"[Debug] Image stats - min: {image.min().item():.2f}, max: {image.max().item():.2f}, mean: {image.mean().item():.2f}")
            print(f"[Info] Model got Image with shape: {image.shape}")

        # >>> Encoding <<<
        #     --------
        feat_cnn = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        if should_print:
            print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = torch.cat([feat_cnn, feat_fno], dim=1)
        # feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        if should_print:
            print(f"feat_comb Shape: {feat_comb.shape}")


        # >>> Decoding <<<
        #     --------
        decoding_cnn = self.cnn_decoder(feat_comb)
        if should_print:
            print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # Decode with SIREN
        # feat_comb_flat = feat_comb.view(batch_size, -1, 512*2)  # Flatten to [B, H*W, 512]
        # [B, C, H, W] -> [B, N, C]
        image_flat = image.view(batch_size, -1, self.in_channels)
        decoding_siren = self.decoder(self.coords, image_flat)
        # Scalars (single values) per pixel to image array -> N = HxW
        decoding_siren = decoding_siren.view(batch_size, self.in_channels, self.image_size, self.image_size)  # [B, N, C] -> [B, 1, H, W]
        if should_print:
            print(f"SIREN Decoder Shape: {decoding_siren.shape}")

        # MLP/CNN take both decodings + the input image and generate the image
        combined = torch.cat([decoding_siren, decoding_cnn, image], dim=1)  # [B, 2C, H, W]
        output = self.output_activation(self.fusion_cnn(combined))

        # MLP whichtakes the fusion and returns the output image
        # output: [B, 64, H, W]
        B, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, 64]
        output = self.mlp_head(output)                     # [B*H*W, in_channels]
        output = output.view(B, H, W, self.in_channels).permute(0, 3, 1, 2)  # [B, in_channels, H, W]
        output = self.output_activation(output)

        # >>> Reshaping <<<
        #     ---------
        # output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        if should_print:
            print(f"Final Output Shape: {output.shape}")
            print(f"[Debug] Output Image stats - min: {output.min().item():.2f}, max: {output.max().item():.2f}, mean: {output.mean().item():.2f}")
            print(f"Finished Forwarding Hexa Wave Net\n{'-'*32}\n")

        self.forward_passes += 1
            
        return output

# Model Type 6
class HexaWaveNetGenerator_4(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.forward_passes = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = SimpleCNNEncoderWithSkips(in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        self.transformer = LatentTransformer(dim=512)

        # self.cnn_decoder = CNNDecoder(in_channels=512, out_channels=64)
        self.decoder = SIRENDecoder(coord_dim=2, latent_dim=960)
        self.output_activation = nn.Tanh()

    def forward(self, image):
        should_print = self.forward_passes == 0

        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # if image.shape[1] == self.in_channels:  # [B, C, H, W] -> [B, H, W, C]
        #     image = image.permute(0, 2, 3, 1)
        

        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net 4")
            print(f"[Debug] Input shape: {image.shape}, min: {image.min():.2f}, max: {image.max():.2f}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        if should_print:
            print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        if should_print:
            print(f"feat_comb Shape: {feat_comb.shape}")

        # >>> Latent Space <<<
        #     ------------
        # Transformer latent attention
        latent_seq = self.transformer(feat_comb)  # [B, 1024, 512]
        B, HW, C = latent_seq.shape
        if should_print:
            print(f"Latent Transformer Shape: {latent_seq.shape}")
        assert HW == 16*16, f"Expected 256 tokens, got {HW}"
        assert C == 512, f"Expected 512 channels, got {C}"

        # >>> Upsample Latent Space <<<
        #     ---------------------
        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        if should_print:
            print(f"Coordinate Shapes: {self.coords.shape}")        

        latent_map = latent_seq.reshape(B, 16, 16, C).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        if should_print:
            print(f"Latent Map Shape: {latent_map.shape}")

        # Upscaling Latent Space
        latent_up = F.interpolate(latent_map, size=(self.image_size, self.image_size), mode='bilinear')  # [B, C, 256, 256]

        # Latent Space in coordinate format for SIREN Decoder
        latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]
        if should_print:
            print(f"Latent Flat Shape: {latent_flat.shape}")

        # >>> Skip Connection <<<
        #     ---------------
        # Process skip features
        skip_flat = process_skips(skips, target_size=(self.image_size, self.image_size))  # [B, N, C_skips]
        if should_print:
            print(f"Skip Flat Shape: {skip_flat.shape}")

        # Combine latent + skips
        full_features = torch.cat([latent_flat, skip_flat], dim=-1)  # [B, N, C_total]
        if should_print:
            print(f"Full Features Shape: {full_features.shape}")

        # >>> Decoding <<<
        #     --------
        # decoding_cnn = self.cnn_decoder(latent_map)
        # if should_print:
        #     print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # Decode with SIREN
        decoding_siren = self.decoder(self.coords, full_features)
        # Scalars (single values) per pixel to image array -> N = HxW
        decoding_siren = decoding_siren.view(batch_size, self.in_channels, self.image_size, self.image_size)  # [B, N, C] -> [B, 1, H, W]
        if should_print:
            print(f"SIREN Decoder Shape: {decoding_siren.shape}")

        # MLP take both decodings + the input image and generate the image
        # combined = torch.cat([decoding_siren, decoding_cnn], dim=1)  # [B, 2C, H, W]
        # fusion = nn.Conv2d(self.in_channels+64, self.in_channels, kernel_size=3, padding=1).to(self.device)
        # output = torch.sigmoid(fusion(combined))
        output = self.output_activation(decoding_siren)

        # >>> Reshaping <<<
        #     ---------
        output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        if should_print:
            print(f"[Debug] Output shape: {output.shape}, min: {output.min():.2f}, max: {output.max():.2f}")
            print(f"FINISHED Forwarding Hexa Wave Net 4\n{'-'*32}")
        self.forward_passes += 1
            
        return output

# Model Type 7
class HexaWaveNetGenerator_5(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.forward_passes = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = SimpleCNNEncoderWithSkips(in_channels, base_channels=64)
        self.fno = FNO(in_channels=512, out_channels=512, width=16, modes1=6, modes2=6)
        self.transformer = LatentTransformer(dim=512)

        self.cnn_decoder = CNNDecoder(in_channels=512, out_channels=64)
        self.fusion_cnn = nn.Conv2d(self.in_channels+64, 64, kernel_size=3, padding=1).to(self.device)

        self.mlp_head = nn.Sequential(
                                nn.Linear(in_features=64, out_features=32, device=self.device, bias=True),
                                nn.Linear(in_features=32, out_features=16, device=self.device, bias=True),
                                nn.Linear(in_features=16, out_features=self.in_channels, device=self.device, bias=True)
                        )
        self.output_activation = nn.Tanh()

    def forward(self, image):
        should_print = self.forward_passes == 0

        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # if image.shape[1] == self.in_channels:  # [B, C, H, W] -> [B, H, W, C]
        #     image = image.permute(0, 2, 3, 1)
        

        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net 4")
            print(f"[Debug] Input shape: {image.shape}, min: {image.min():.2f}, max: {image.max():.2f}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")
        feat_fno = self.fno(feat_cnn)   # [B, 512, 16, 16]
        if should_print:
            print(f"feat_fno Shape: {feat_fno.shape}")
        feat_comb = feat_cnn + feat_fno  # Combine CNN and FNO features (Skip Connecion)
        if should_print:
            print(f"feat_comb Shape: {feat_comb.shape}")

        # >>> Latent Space <<<
        #     ------------
        # Transformer latent attention
        latent_seq = self.transformer(feat_comb)  # [B, 1024, 512]
        B, HW, C = latent_seq.shape
        if should_print:
            print(f"Latent Transformer Shape: {latent_seq.shape}")
        assert HW == 16*16, f"Expected 256 tokens, got {HW}"
        assert C == 512, f"Expected 512 channels, got {C}"

        # >>> Upsample Latent Space <<<
        #     ---------------------
        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        if should_print:
            print(f"Coordinate Shapes: {self.coords.shape}")        

        latent_map = latent_seq.reshape(B, 16, 16, C).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        if should_print:
            print(f"Latent Map Shape: {latent_map.shape}")

        # # Upscaling Latent Space
        # latent_up = F.interpolate(latent_map, size=(self.image_size, self.image_size), mode='bilinear')  # [B, C, 256, 256]

        # # Latent Space in coordinate format for SIREN Decoder
        # latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]
        # if should_print:
        #     print(f"Latent Flat Shape: {latent_flat.shape}")

        # # >>> Skip Connection <<<
        # #     ---------------
        # # Process skip features
        # skip_flat = process_skips(skips, target_size=(self.image_size, self.image_size))  # [B, N, C_skips]
        # if should_print:
        #     print(f"Skip Flat Shape: {skip_flat.shape}")

        # # Combine latent + skips
        # full_features = torch.cat([latent_flat, skip_flat], dim=-1)  # [B, N, C_total]
        # if should_print:
        #     print(f"Full Features Shape: {full_features.shape}")

        # >>> Decoding <<<
        #     --------
        decoding_cnn = self.cnn_decoder(latent_map)
        if should_print:
            print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # MLP/CNN take decoding + the input image and generate the image
        combined = torch.cat([decoding_cnn, image], dim=1)  # [B, 2C, H, W]
        output = self.output_activation(self.fusion_cnn(combined))

        # MLP whichtakes the fusion and returns the output image
        # output: [B, 64, H, W]
        B, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, 64]
        output = self.mlp_head(output)                     # [B*H*W, in_channels]
        output = output.view(B, H, W, self.in_channels).permute(0, 3, 1, 2)  # [B, in_channels, H, W]

        # >>> Reshaping <<<
        #     ---------
        output = output.reshape(batch_size, self.in_channels, self.image_size, self.image_size)

        if should_print:
            print(f"[Debug] Output shape: {output.shape}, min: {output.min():.2f}, max: {output.max():.2f}")
            print(f"FINISHED Forwarding Hexa Wave Net 4\n{'-'*32}")
        self.forward_passes += 1
            
        return output

# Model Type 8
class HexaWaveNetGenerator_6(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, image_size=256):
        super().__init__()

        self.forward_passes = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.in_channels = in_channels
        self.coords = get_normalized_grid(H=image_size, W=image_size, device=self.device)

        self.encoder = SimpleCNNEncoderWithSkips(in_channels, base_channels=64)
        self.transformer = LatentTransformer(dim=512)

        self.cnn_decoder = CNNDecoder(in_channels=512, out_channels=64)
        self.fusion_cnn = nn.Conv2d(self.in_channels+64, 64, kernel_size=3, padding=1).to(self.device)

        self.mlp_head = nn.Sequential(
                                nn.Linear(in_features=64, out_features=32, device=self.device, bias=True),
                                nn.Linear(in_features=32, out_features=16, device=self.device, bias=True),
                                nn.Linear(in_features=16, out_features=self.in_channels, device=self.device, bias=True)
                        )

        cnn_head_in = self.in_channels+64+64*2+64*4
        self.cnn_head_1  = nn.Conv2d(in_channels=cnn_head_in, out_channels=cnn_head_in//2, device=self.device, bias=True, kernel_size=1, stride=1)
        self.cnn_head_2  = nn.Conv2d(in_channels=cnn_head_in//2, out_channels=cnn_head_in//4, device=self.device, bias=True, kernel_size=1, stride=1)
        self.cnn_head_3  = nn.Conv2d(in_channels=cnn_head_in//4, out_channels=cnn_head_in//8, device=self.device, bias=True, kernel_size=1, stride=1)
        self.cnn_head_4  = nn.Conv2d(in_channels=cnn_head_in//8, out_channels=self.in_channels, device=self.device, bias=True, kernel_size=1, stride=1)

        self.output_activation = nn.Tanh()

    def forward(self, image):
        should_print = self.forward_passes == 0

        # make right dimensions: [B, C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
            image = image.unsqueeze(3)
        elif image.dim() == 3:
            image = image.unsqueeze(0)

        # if image.shape[1] == self.in_channels:  # [B, C, H, W] -> [B, H, W, C]
        #     image = image.permute(0, 2, 3, 1)
        

        batch_size = image.shape[0]

        if should_print:
            print(f"\n{'-'*32}\nForwarding Hexa Wave Net 6")
            print(f"[Debug] Input shape: {image.shape}, min: {image.min():.2f}, max: {image.max():.2f}")

        # >>> Encoding <<<
        #     --------
        feat_cnn, skips = self.encoder(image)  # [B, 512, 16, 16]
        if should_print:
            print(f"feat_cnn Shape: {feat_cnn.shape}")

        # >>> Latent Space <<<
        #     ------------
        # Transformer latent attention
        latent_seq = self.transformer(feat_cnn)  # [B, 256, 512]
        B, HW, C = latent_seq.shape
        if should_print:
            print(f"Latent Transformer Shape: {latent_seq.shape}")
        assert HW == 16*16, f"Expected 256 tokens, got {HW}"
        assert C == 512, f"Expected 512 channels, got {C}"

        # >>> Upsample Latent Space <<<
        #     ---------------------
        # Upsample latent features to match coordinates (assumes coords is Nx2 for 256x256)
        if should_print:
            print(f"Coordinate Shapes: {self.coords.shape}")        

        latent_map = latent_seq.reshape(B, 16, 16, C).permute(0, 3, 1, 2)  # [B, C, 16, 16]
        if should_print:
            print(f"Latent Map Shape: {latent_map.shape}")

        # # Upscaling Latent Space
        # latent_up = F.interpolate(latent_map, size=(self.image_size, self.image_size), mode='bilinear')  # [B, C, 256, 256]

        # # Latent Space in coordinate format for SIREN Decoder
        # latent_flat = rearrange(latent_up, 'b c h w -> b (h w) c')  # [B, N, C]
        # if should_print:
        #     print(f"Latent Flat Shape: {latent_flat.shape}")

        # >>> Skip Connection <<<
        #     ---------------
        # Process skip features
        upsampled_skips = process_skips(skips, target_size=(self.image_size, self.image_size), flatten=False)  # [B, C_total, 256, 256]
        if should_print:
            print(f"Upsampled Skips Shape: {upsampled_skips.shape}")

        # # Combine latent + skips
        # full_features = torch.cat([latent_flat, skip_flat], dim=-1)  # [B, N, C_total]
        # if should_print:
        #     print(f"Full Features Shape: {full_features.shape}")

        # >>> Decoding <<<
        #     --------
        decoding_cnn = self.cnn_decoder(latent_map)
        if should_print:
            print(f"CNN Decoder Shape: {decoding_cnn.shape}")

        # MLP/CNN take decoding + the input image and generate the image
        combined = torch.cat([decoding_cnn, image], dim=1)  # [B, 2C, H, W]
        output = self.output_activation(self.fusion_cnn(combined))

        # MLP whichtakes the fusion and returns the output image
        # output: [B, 64, H, W]
        B, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, 64]
        output = self.mlp_head(output)                     # [B*H*W, in_channels]
        output = output.view(B, H, W, self.in_channels).permute(0, 3, 1, 2)  # [B, in_channels, H, W]
        if should_print:
            print(f"MLP Head Shape: {output.shape}")

        # CNN Head
        cnn_head_1 = self.cnn_head_1(torch.cat([output, upsampled_skips], dim=1))
        cnn_head_2 = self.cnn_head_2(cnn_head_1)
        cnn_head_3 = self.cnn_head_3(cnn_head_2)
        cnn_head_4 = self.cnn_head_4(cnn_head_3)

        if should_print:
            print(f"CNN Head 1 Shape: {cnn_head_1.shape}")
            print(f"CNN Head 2 Shape: {cnn_head_2.shape}")
            print(f"CNN Head 3 Shape: {cnn_head_3.shape}")
            print(f"CNN Head 4 Shape: {cnn_head_4.shape}")

        # >>> Reshaping <<<
        #     ---------
        output = cnn_head_4.reshape(batch_size, self.in_channels, self.image_size, self.image_size)
        output = self.output_activation(output)

        if should_print:
            print(f"[Debug] Output shape: {output.shape}, min: {output.min():.2f}, max: {output.max():.2f}")
            print(f"FINISHED Forwarding Hexa Wave Net 6\n{'-'*32}")
        self.forward_passes += 1
            
        return output

#  SIREN Only Model
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENImageTranslator(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, omega_0=30):
        super().__init__()
        self.forward_passes = 0

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers):
            layers += [SineLayer(hidden_features, hidden_features, omega_0=omega_0)]
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_features, out_features)

        self.output_activation = nn.Tanh()

    def forward(self, x):
        should_print = self.forward_passes == 0
        # Handle input shape: [B, C, H, W] → [B, H, W, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            x = x.unsqueeze(3)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        if should_print:
            print(f"\n{'-'*32}\nForwarding SIREN")
            print(f"[Debug] Input shape: {x.shape}, min: {x.min():.2f}, max: {x.max():.2f}")
        
        batch_size = x.shape[0]
        channels = x.shape[1]
        image_size = x.shape[2]

        if should_print:
            print(f"Input Image Shape: {x.shape}")

        # x = x.permute(0,2,3,1).reshape(-1, 3)
        x = x.view(batch_size, -1, channels)
        if should_print:
            print(f"Reshaped Input Image Shape: {x.shape}")

        output = self.final(self.net(x))
        if should_print:
            print(f"Output Shape: {output.shape}")

        output = output.view(batch_size, channels, image_size, image_size)
        output = self.output_activation(output)

        if should_print:
            print(f"[Debug] Output shape: {output.shape}, min: {output.min():.2f}, max: {output.max():.2f}")
            print(f"FINISHED Forwarding SIREN\n{'-'*32}")
        self.forward_passes += 1

        return output

# FNO Only
class Spectralizer(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.forward_passes = 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_x, modes_y, 2))

    def compl_mul2d(self, input, weights):
        # real = input[..., 0] * weights[..., 0] - input[..., 1] * weights[..., 1]
        # imag = input[..., 0] * weights[..., 1] + input[..., 1] * weights[..., 0]
        # return torch.stack([real, imag], dim=-1)

        # input: (B, C_in, X, Y, 2)
        # weights: (C_in, C_out, X, Y, 2)
        # output: (B, C_out, X, Y, 2)
        return torch.einsum("bixyq,ioxyq->boxyq", input, weights)

    def forward(self, x):
        should_print = self.forward_passes == 0
        batchsize, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm='forward')
        out_ft = torch.zeros(batchsize, self.out_channels, height, width // 2 + 1, 2, device=x.device)

        if should_print:
            print(f"Spectralizer - xft shape: {x_ft.shape}")
            print(f"Spectralizer - out_ft shape: {out_ft.shape}")

        x_ft_realimag = torch.stack([x_ft.real, x_ft.imag], dim=-1)
        if should_print:
            print(f"Spectralizer - x_ft_realimag shape: {x_ft_realimag.shape}")
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.compl_mul2d(
            x_ft_realimag[:, :, :self.modes_x, :self.modes_y],
            self.weights
        )

        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_complex, s=(height, width), norm='forward')

        if should_print:
            print(f"Spectralizer - output shape: {x.shape}")

        self.forward_passes += 1

        return x

class FNO2D(nn.Module):
    def __init__(self, modes_x, modes_y, width, in_channels):
        super().__init__()
        self.forward_passes = 0
        self.in_channels = in_channels
        self.width = width

        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for positional grid

        self.conv0 = Spectralizer(width, width, modes_x, modes_y)
        self.conv1 = Spectralizer(width, width, modes_x, modes_y)
        self.conv2 = Spectralizer(width, width, modes_x, modes_y)
        self.conv3 = Spectralizer(width, width, modes_x, modes_y)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.output_activation = nn.Tanh()

    def forward(self, x):
        should_print = self.forward_passes == 0

        # Handle input shape: [B, C, H, W] → [B, H, W, C]
        if x.dim() == 2:
            x = x.unsqueeze(0)
            x = x.unsqueeze(3)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        if should_print:
            print(f"\n{'-'*32}\nForwarding Fourier Neural Operator")
            print(f"[Debug] Input shape: {x.shape}, min: {x.min():.2f}, max: {x.max():.2f}")

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        grid = self.get_grid(x.shape, x.device)  # [B, H, W, 2]
        if should_print:
            print(f"Grid Shape: {grid.shape}")

        x = torch.cat([x, grid], dim=-1)  # [B, H, W, in_channels + 2]
        if should_print:
            print(f"Input + Grid Shape: {x.shape}")

        x = self.fc0(x)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
        if should_print:
            print(f"Post-FC0 Shape: {x.shape}")

        x = self.conv0(x) + self.w0(x)
        x = self.conv1(x) + self.w1(x)
        x = self.conv2(x) + self.w2(x)
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)  # [B, H, W, 1]

        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W] 

        x = self.output_activation(x)

        if should_print:
            print(f"[Debug] Output shape: {x.shape}, min: {x.min():.2f}, max: {x.max():.2f}")
            print(f"FINISHED Forwarding Fourier Neural Operator\n{'-'*32}")
        self.forward_passes += 1
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, _ = shape
        gridx = torch.linspace(0, 1, size_x, device=device)
        gridy = torch.linspace(0, 1, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')  # [H, W]
        grid = torch.stack((gridx, gridy), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(batchsize, 1, 1, 1)  # [B, H, W, 2]
        return grid



def get_normalized_grid(H=256, W=256, device="cuda"):
    # Creates a normalized meshgrid from -1 to 1
    y = torch.linspace(-1, 1, H).to(device)
    x = torch.linspace(-1, 1, W).to(device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    coords = coords.view(-1, 2)  # [H*W, 2]
    return coords


# Tranformer models
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=512, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, emb_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth=8, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(*[TransformerBlock(**kwargs) for _ in range(depth)])

    def forward(self, x):
        return self.layers(x)

class TransformerImageTranslator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, emb_dim=512, depth=8, heads=8, mlp_dim=1024):
        super().__init__()
        self.forward_passes = 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.encoder = TransformerEncoder(depth, dim=emb_dim, heads=heads, mlp_dim=mlp_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, patch_size * patch_size),
            nn.Unflatten(2, (patch_size, patch_size)),
        )
        # self.out_conv = nn.ConvTranspose2d(1, 1, kernel_size=patch_size, stride=patch_size)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        should_print = self.forward_passes == 0
        if x.dim() == 2:
            x = x.unsqueeze(0)
            x = x.unsqueeze(3)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        if should_print:
            print(f"\n{'-'*32}\nForwarding Transformer")
            print(f"[Debug] Input shape: {x.shape}, min: {x.min():.2f}, max: {x.max():.2f}")

        B = x.shape[0]

        patches = self.patch_embedding(x)               # (B, N, D)
        if should_print:
            print(f"Patch Shape: {patches.shape}")

        features = self.encoder(patches)                # (B, N, D)
        if should_print:
            print(f"Encoding Shape: {features.shape}")

        decoded = self.decoder(features)                    # (B, H, H/patch, W/patch)  [1, 256, 16, 16]
        if should_print:
            print(f"Decoding Shape: {decoded.shape}")

        grid_size = self.grid_size
        out = decoded.view(B, grid_size, grid_size, self.patch_size, self.patch_size)
        out = out.permute(0, 1, 3, 2, 4).contiguous()  # (B, H/P, P, W/P, P)
        out = out.view(B, 1, self.img_size, self.img_size)  # (B, 1, H, W)

        # out = self.out_conv(out) 
        out = self.output_activation(out) 
        
        if should_print:
            print(f"[Debug] Output shape: {out.shape}, min: {out.min():.2f}, max: {out.max():.2f}")
            print(f"FINISHED Transformer\n{'-'*32}")
        self.forward_passes += 1                      # (B, 1, 256, 256)

        return out



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
            parser.add_argument('--lambda_GAN', type=float, default=2.0, help='weight for GAN loss')
            parser.add_argument('--lambda_ssmi', type=float, default=100.0, help='weight for SSMI loss')
            parser.add_argument('--lambda_edge', type=float, default=100.0, help='weight for Edge loss')
            # parser.add_argument('--gan_activate_epoch', type=int, default=50, help='Epoch on which GAN loss is disabled')
            # parser.add_argument('--gan_disable_epoch', type=int, default=100, help='Epoch after which GAN loss is disabled')
            parser.add_argument('--wgangp', action='store_true', help='Should use WGAN-GP')

        parser.add_argument('--model_type', type=int, default=1, help='Decides the exact HexaWaveNet model (see in the model.py for the different models).')

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
        # FIXME + change me
        self.opt.model_type = 8
        if self.opt.model_type == 1:
            hexa_wave_net = HexaWaveNetGenerator_1(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 2:
            hexa_wave_net = HexaWaveNetGenerator_2(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 3:
            hexa_wave_net = HexaWaveNetGenerator_3(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 4:
            hexa_wave_net = SIRENImageTranslator(in_features=opt.input_nc, out_features=opt.input_nc)
        elif self.opt.model_type == 5:
            hexa_wave_net = FNO2D(modes_x=12, modes_y=12, width=32, in_channels=opt.input_nc)
        elif self.opt.model_type == 6:
            hexa_wave_net = HexaWaveNetGenerator_4(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 7:
            hexa_wave_net = HexaWaveNetGenerator_5(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 8:
            hexa_wave_net = HexaWaveNetGenerator_6(in_channels=opt.input_nc, latent_dim=512, image_size=opt.crop_size)
        elif self.opt.model_type == 9:
            hexa_wave_net = TransformerImageTranslator(img_size=opt.crop_size, patch_size=16, in_channels=opt.input_nc, emb_dim=512, depth=8, heads=8, mlp_dim=1024)
        else:
            raise ValueError(f"Hexa Wave Net does not have the type: {self.opt.model_type}")
        self.netG = networks.init_net(hexa_wave_net, 
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
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.current_epoch = 0

        # currently deactivated
        self.epochs_with_gan = 0
        self.gan_active = True
        self.forward_passes = 0

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

        # if new_epoch:
        #     if self.gan_active:
        #         self.epochs_with_gan += 1
        #         print("[Loss-System] Epochs with GAN Mode increased")

        #         if self.epochs_with_gan >= self.opt.gan_disable_epoch:
        #             print("[Loss-System] Deactivated GAN Loss Mode")
        #             self.gan_active = False
        
        #     if not self.gan_active:
        #         if self.epochs_with_gan < self.opt.gan_disable_epoch and self.current_epoch >= self.opt.gan_activate_epoch:
        #             print("[Loss-System] Activated GAN Loss Mode")
        #             self.gan_active = True

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

    # New #
    def backward_D(self):
        """
        Calculate GAN loss for the discriminator.

        Wasserstein loss with gradient penalty.
        """
        fake_AB = torch.cat((self.real_A, self.fake_B), dim=1)
        real_AB = torch.cat((self.real_A, self.real_B), dim=1)

        pred_fake = self.netD(fake_AB.detach())
        pred_real = self.netD(real_AB)

        if self.opt.wgangp:
            # WGAN loss
            self.loss_D_real = -pred_real.mean()
            self.loss_D_fake = pred_fake.mean()

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
        """
        Generator loss with optional GAN loss
        """
        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # GAN Loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        if self.opt.wgangp:
            self.loss_G_GAN = -pred_fake.mean()
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # combine loss and calculate gradients

        # SSMI Loss
        loss_ssmi = 1 - pytorch_msssim.ssim(self.fake_B, self.real_B, data_range=1.0) 

        # Edge Loss
        loss_edges = gradient_loss(self.fake_B, self.real_B)
        
        # Combine all losses
        self.loss_G = self.loss_G_L1 * self.opt.lambda_L1 + \
                      self.loss_G_GAN * self.opt.lambda_GAN + \
                      loss_ssmi * self.opt.lambda_ssmi + \
                      loss_edges * self.opt.lambda_edge

        self.loss_G.backward()

    def optimize_parameters(self, check_vanishing_gradients=False):
        self.forward()

        # Discrimantor
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            self.backward_D()
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
        self.optimizer_D.step()

        # Generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optimizer_G.step()

        # Test info log
        if check_vanishing_gradients:
            mean_gradient = []
            critical_parts = dict()
            for name, param in self.netG.named_parameters():
                if param.grad is not None:
                    mean_gradient += [param.grad.abs().mean().item()]    # .item() gets python value
                    if param.grad.abs().mean().item() < 0.01:
                        critical_parts[name] = param.grad.abs().mean().item()
            mean_gradient = numpy.array(mean_gradient)
            if mean_gradient.size > 0:
                print(f"[Vanishing Gradient Check] mean: {mean_gradient.mean():.2f}, min: {mean_gradient.min():.2f}, max: {mean_gradient.max():.2f}")
                print("    Critical:"+ "".join(['\n        - Param: '+name+', Mean Gradient: '+str(value)for name, value in critical_parts.items()]))
            else:
                print("[Vanishing Gradient Check] No gradients available.")

        if check_vanishing_gradients:
            criticals = ""
            for name, param in self.netG.named_parameters():
                if not param.requires_grad or param.requires_grad == False:
                    criticals += f"\n      - {name} -> grad activated: {param.requires_grad}"
            if len(criticals) > 0:
                print("Parameter Gradient Activation Check:")
                print(criticals)

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
def gradient_loss(pred, target):
    sobel = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
    sobel.weight.data = torch.tensor([
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
    ], dtype=torch.float32)
    sobel.requires_grad_(False)
    sobel = sobel.to("cuda")

    pred_grad = sobel(pred)
    target_grad = sobel(target)
    return F.l1_loss(pred_grad, target_grad)

def compute_gradient_penalty(D, real_samples, fake_samples, device, lambda_gp=10.0):
    """
    Calculates the gradient penalty loss for WGAN GP
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

    





