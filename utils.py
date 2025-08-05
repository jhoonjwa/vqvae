import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time
import os
import numpy as np
from datasets.file_dataset import FileDataset
from datasets.glb_dataset import GLBDataset


def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block():
    from datasets.block import BlockDataset
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val

def load_latent_block():
    from datasets.block import LatentBlockDataset
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def load_file(data_path):
    """Load a generic dataset stored as a NumPy file."""
    if data_path is None:
        raise ValueError('data_path must be provided for FILE dataset')
    data = np.load(data_path)
    n = len(data)
    split = int(0.9 * n)
    train = FileDataset(data_path)
    val = FileDataset(data_path)
    # split datasets without copying large arrays
    train.data = train.data[:split]
    val.data = val.data[split:]
    return train, val


def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_glb(glb_dir, val_size=5):
    """Load GLB dataset and split into train/validation sets."""
    full_dataset = GLBDataset(glb_dir)
    
    # Split dataset into train/validation
    total_size = len(full_dataset)
    train_size = total_size - val_size
    
    if train_size <= 0:
        raise ValueError(f"Not enough samples for training. Total: {total_size}, Val size: {val_size}")
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"GLB Dataset - Total: {total_size}, Train: {train_size}, Val: {val_size}")
    
    return train_dataset, val_dataset

def load_data_and_data_loaders(dataset, batch_size, data_path=None, glb_dir=None, val_size=5):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)

        x_train_var = np.var(training_data.data)
    elif dataset == 'FILE':
        training_data, validation_data = load_file(data_path)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.data)
    elif dataset == 'GLB':
        training_data, validation_data = load_glb(glb_dir, val_size)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        # For GLB data, we'll compute variance from a sample
        sample_data = []
        for i, (x, _) in enumerate(training_loader):
            sample_data.append(x.numpy())
            if i >= 5:  # Use first 5 batches to estimate variance
                break
        if sample_data:
            x_train_var = np.var(np.concatenate(sample_data, axis=0))
        else:
            x_train_var = 1.0  # fallback
    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10, BLOCK, LATENT_BLOCK, FILE and GLB datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
