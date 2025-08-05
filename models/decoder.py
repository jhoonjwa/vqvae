
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock2d(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, self.out_channels)
        self.conv1 = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip_connection = nn.Conv2d(channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()
    
    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class UpsampleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """
    2D decoder with upsampling to convert [32, 32, 36] -> [512, 512, 9]
    
    Input: [batch, 36, 32, 32]
    Output: [batch, 9, 512, 512]
    """

    def __init__(self, latent_channels=36, out_channels=9, num_res_blocks=2, channels=[512, 256, 128, 64, 32], num_res_blocks_middle=2):
        super(Decoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        
        # Input projection: 36 -> 512 channels
        self.input_layer = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        
        # Middle blocks at the deepest level
        self.middle_block = nn.Sequential(*[
            ResBlock2d(channels[0], channels[0]) for _ in range(num_res_blocks_middle)
        ])
        
        # Decoder blocks with upsampling
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            # Add residual blocks for this level
            self.blocks.extend([
                ResBlock2d(ch, ch) for _ in range(num_res_blocks)
            ])
            # Add upsampling block (except for the last level)
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock2d(ch, channels[i+1])
                )

        # Output layer
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], out_channels, 3, padding=1)
        )

    def forward(self, x):
        # Input: [batch, 36, 32, 32]
        h = self.input_layer(x)  # [batch, 512, 32, 32]
        
        # Middle blocks
        h = self.middle_block(h)
        
        # Pass through decoder blocks with upsampling
        # 32 -> 64 -> 128 -> 256 -> 512
        for block in self.blocks:
            h = block(h)
        
        # Output projection
        h = self.out_layer(h)  # [batch, 9, 512, 512]
        return h


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
