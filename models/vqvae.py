
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, in_channels=9, latent_channels=36, n_embeddings=8192, embedding_dim=36, beta=0.25, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space: [512, 512, 9] -> [32, 32, 36]
        self.encoder = Encoder(
            in_channels=in_channels, 
            latent_channels=latent_channels,
            num_res_blocks=2,
            channels=[32, 64, 128, 256, 512],
            num_res_blocks_middle=2
        )
        
        # pre-quantization convolution (if embedding_dim != latent_channels)
        if embedding_dim != latent_channels:
            self.pre_quantization_conv = nn.Conv2d(
                latent_channels, embedding_dim, kernel_size=1, stride=1)
        else:
            self.pre_quantization_conv = nn.Identity()
            
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
            
        # decode the discrete latent representation: [32, 32, 36] -> [512, 512, 9]
        self.decoder = Decoder(
            latent_channels=embedding_dim,
            out_channels=in_channels,
            num_res_blocks=2,
            channels=[512, 256, 128, 64, 32],
            num_res_blocks_middle=2
        )

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        # Input: [batch, 9, 512, 512]
        z_e = self.encoder(x)  # [batch, 36, 32, 32]

        z_e = self.pre_quantization_conv(z_e)  # [batch, 36, 32, 32]
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)  # [batch, 36, 32, 32]
        x_hat = self.decoder(z_q)  # [batch, 9, 512, 512]

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
