from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import torch
import pytorch_lightning as L


# split the train set into two
seed = torch.Generator().manual_seed(42)



class CVAE_MNIST_Data(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['train_batch_size']
        self.val_batch_size = config['val_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.data_dir = config['data_dir']

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        # Unlike in example 02, we do NOT transform the image into a vector, since we wan tto apply 2D convolutions
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5),(1.0)),#(0.1307,), (0.3081,)),
                                ])

        self.train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.val_set_size = len(self.train_dataset) - self.train_set_size
    
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_set_size, self.val_set_size])
    
        self.test_dataset = datasets.MNIST(root= self.data_dir, train=False, download=True, transform=transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=8, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=8, persistent_workers=True, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=8, persistent_workers=True, shuffle=False)
    
    #def state_dict(self):
    #    # track whatever you want here
    #    state = {"current_train_batch_index": self.current_train_batch_index}
    #    return state

    #def load_state_dict(self, state_dict):
    #    # restore the state based on what you tracked in (def state_dict)
    #    self.current_train_batch_index = state_dict["current_train_batch_index"]

    # def transfer_batch_to_device(self, batch, device, dataloader_idx)
    # This lets you specify transformations when transfering data to a device
    # more hooks are available here (mostly useful for multi node setups): https://lightning.ai/docs/pytorch/stable/data/datamodule.html