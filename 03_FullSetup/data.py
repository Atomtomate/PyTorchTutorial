from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
import numpy as np

tf = Compose([ToTensor(), Lambda(torch.flatten)])
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=tf)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=tf)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
val_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)

train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size], generator=seed)
train_loader = utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8, persistent_workers=True, shuffle=True)
test_loader = utils.data.DataLoader(test_set, batch_size=1, num_workers=8, persistent_workers=True, shuffle=False)
val_loader = utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=8, persistent_workers=True, shuffle=False)

num_data_workers
class Dataset_CVAE_MNIST(Dataset):
    def __init__(self, config):
        # Unlike in example 02, we do NOT transform the image into a vector, since we wan tto apply 2D convolutions
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

        train_dataset = datasets.MNIST(root ='data', train=True, download=True, transform=transform)
        # use 20% of training data for validation
        train_set_size = int(len(train_dataset) * 0.8)
        val_set_size = len(train_dataset) - train_set_size
        train_dataset, val_dataset = data.random_split(train_set, [train_set_size, val_set_size])
        test_dataset = datasets.MNIST(root = 'data', train=False, transform=transform, num_workers=8, persistent_workers=True)

        train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], num_workers=8, persistent_workers=True, shuffle=True)
        val_loader  = DataLoader(train_dataset, batch_size=config['train_batch_size'], num_workers=8, persistent_workers=True, shuffle=True)



        test_loader = DataLoader( test_dataset, batch_size=config['test_batch_size'], shuffle=False)


    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        return self.data_in[idx], self.data_target[idx]
