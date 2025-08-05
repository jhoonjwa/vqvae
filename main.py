import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from tqdm import tqdm

parser = argparse.ArgumentParser()

'''
CUDA_VISIBLE_DEVICES=0 python main.py --dataset GLB --glb_dir /home/ubuntu/jonghoon/mesh2mesh/vqvae/datasets/glbs/under_20k --val_size 5
  --early_stopping_patience 3 --val_freq 0.1
'''

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_updates", type=int, default=25000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=1024)
parser.add_argument("--n_embeddings", type=int, default=8192)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--dataset",  type=str, default='FILE')
parser.add_argument("--in_channels", type=int, default=9)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--glb_dir", type=str, default="/home/ubuntu/jonghoon/mesh2mesh/vqvae/datasets/glbs/under_20k")
parser.add_argument("--val_size", type=int, default=5, help="Number of samples to use for validation")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
parser.add_argument("--val_freq", type=float, default=0.1, help="Validation frequency as fraction of epoch")

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size, args.data_path, args.glb_dir, args.val_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(in_channels=args.in_channels, 
              latent_channels=args.embedding_dim, 
              n_embeddings=args.n_embeddings, 
              embedding_dim=args.embedding_dim, 
              beta=args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'val_losses': [],
    'best_val_loss': float('inf'),
    'patience_counter': 0,
}


def validate():
    """Run validation and return average validation loss"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for x, _ in validation_loader:
            x = x.to(device)
            embedding_loss, x_hat, _ = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            val_loss = recon_loss + embedding_loss
            val_losses.append(val_loss.cpu().item())
    
    model.train()
    return np.mean(val_losses)

def train():
    # Calculate validation frequency in terms of updates
    samples_per_epoch = len(training_data)
    updates_per_epoch = samples_per_epoch // args.batch_size
    val_interval = max(1, int(updates_per_epoch * args.val_freq))
    
    print(f"Training with {len(training_data)} samples, {len(validation_data)} validation samples")
    print(f"Updates per epoch: {updates_per_epoch}, Validation every {val_interval} updates")

    train_iterator = iter(training_loader)
    
    # Create progress bar
    pbar = tqdm(range(args.n_updates), desc="Training", unit="update")
    
    for i in pbar:
        try:
            x, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(training_loader)
            x, _ = next(train_iterator)
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        # Validation and early stopping check
        if i % val_interval == 0 and i > 0:
            val_loss = validate()
            results["val_losses"].append(val_loss)
            
            tqdm.write(f'Update #{i}, Val Loss: {val_loss:.6f}, Best: {results["best_val_loss"]:.6f}')
            
            # Early stopping logic
            if val_loss < results["best_val_loss"]:
                results["best_val_loss"] = val_loss
                results["patience_counter"] = 0
                if args.save:
                    # Save best model
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename + "_best")
            else:
                results["patience_counter"] += 1
                
            if results["patience_counter"] >= args.early_stopping_patience:
                tqdm.write(f'Early stopping triggered after {results["patience_counter"]} validation steps without improvement')
                break
    
        # Update progress bar with recent metrics
        if len(results["loss_vals"]) > 0:
            recent_loss = np.mean(results["loss_vals"][-10:])
            recent_recon = np.mean(results["recon_errors"][-10:])
            recent_perp = np.mean(results["perplexities"][-10:])
            pbar.set_postfix({
                'Loss': f'{recent_loss:.4f}',
                'Recon': f'{recent_recon:.2e}',
                'Perp': f'{recent_perp:.1f}',
                'Best Val': f'{results["best_val_loss"]:.4f}' if results["best_val_loss"] != float('inf') else 'N/A'
            })

        if i % args.log_interval == 0:
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            tqdm.write(f'Update #{i}, Recon Error: {np.mean(results["recon_errors"][-args.log_interval:]):.2e}, '
                      f'Loss: {np.mean(results["loss_vals"][-args.log_interval:]):.4f}, '
                      f'Perplexity: {np.mean(results["perplexities"][-args.log_interval:]):.1f}')
    
    pbar.close()


if __name__ == "__main__":
    train()
