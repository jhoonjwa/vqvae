import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        b, h, w, _ = z.shape
        z_flattened = z.view(-1, self.e_dim)

        # compute distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
             torch.sum(self.embedding.weight ** 2, dim=1) -
             2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        # find closest encodings
        encoding_indices = torch.argmin(d, dim=1)

        # get quantized latent vectors using embedding lookup
        z_q = self.embedding(encoding_indices).view(b, h, w, self.e_dim)

        # compute loss for embedding and commitment
        loss = torch.mean((z.detach() - z_q) ** 2) + \
            self.beta * torch.mean((z - z_q.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity using counts to avoid large one-hot matrices
        encodings_count = torch.bincount(encoding_indices, minlength=self.n_e).float()
        e_mean = encodings_count / encodings_count.sum()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        encoding_indices = encoding_indices.view(b, h, w)

        return loss, z_q, perplexity, encodings_count, encoding_indices
