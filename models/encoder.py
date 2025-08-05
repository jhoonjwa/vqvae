
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


class DownsampleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """
    2D encoder with downsampling to convert [512, 512, 9] -> [32, 32, 36]
    
    Input: [batch, 9, 512, 512]
    Output: [batch, 36, 32, 32]
    """

    def __init__(self, in_channels=9, latent_channels=36, num_res_blocks=2, channels=[32, 64, 128, 256, 512], num_res_blocks_middle=2):
        super(Encoder, self).__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        
        # Input projection: 9 -> 64 channels
        self.input_layer = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder blocks with downsampling
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            # Add residual blocks for this level
            self.blocks.extend([
                ResBlock2d(ch, ch) for _ in range(num_res_blocks)
            ])
            # Add downsampling block (except for the last level)
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock2d(ch, channels[i+1])
                )
        
        # Middle blocks at the deepest level
        self.middle_block = nn.Sequential(*[
            ResBlock2d(channels[-1], channels[-1]) for _ in range(num_res_blocks_middle)
        ])

        # Output layer to get latent_channels
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], latent_channels, 3, padding=1)
        )

    def forward(self, x):
        # Input: [batch, 9, 512, 512]
        h = self.input_layer(x)  # [batch, 64, 512, 512]
        
        # Pass through encoder blocks with downsampling
        # 512 -> 256 -> 128 -> 64 -> 32
        for block in self.blocks:
            h = block(h)
        
        # Middle blocks at [batch, 512, 32, 32]
        h = self.middle_block(h)
        
        # Output projection
        h = self.out_layer(h)  # [batch, 36, 32, 32]
        return h


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('Encoder out shape:', encoder_out.shape)
