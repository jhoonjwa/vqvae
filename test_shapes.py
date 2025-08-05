#!/usr/bin/env python3

import torch
import sys
import os

# Add models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.vqvae import VQVAE

def test_vqvae_shapes():
    """Test VQ-VAE with the required input/output shapes"""
    
    # Create model
    model = VQVAE(
        in_channels=9,
        latent_channels=36, 
        n_embeddings=8192,
        embedding_dim=36,
        beta=0.25
    )
    
    # Create test input: [batch, 9, 512, 512]
    batch_size = 2
    x = torch.randn(batch_size, 9, 512, 512)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Test encoder
        z_e = model.encoder(x)
        print(f"Encoder output shape: {z_e.shape}")
        
        # Test full forward pass
        embedding_loss, x_hat, perplexity = model(x)
        print(f"Reconstruction shape: {x_hat.shape}")
        print(f"Embedding loss: {embedding_loss.item():.4f}")
        print(f"Perplexity: {perplexity.item():.4f}")
        
        # Verify shapes
        expected_latent_shape = (batch_size, 36, 32, 32)
        expected_output_shape = (batch_size, 9, 512, 512)
        
        assert z_e.shape == expected_latent_shape, f"Expected latent shape {expected_latent_shape}, got {z_e.shape}"
        assert x_hat.shape == expected_output_shape, f"Expected output shape {expected_output_shape}, got {x_hat.shape}"
        
        print("✅ All shape tests passed!")
        print(f"✅ Input: [512, 512, 9] -> Latent: [32, 32, 36] -> Output: [512, 512, 9]")

if __name__ == "__main__":
    test_vqvae_shapes()