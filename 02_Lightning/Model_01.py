import os
from torch import optim, nn, utils, Tensor, flatten
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

class AutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()

        # Activation function as class variable
        # Later we will define things like this through a config file and then define the appropriate activation here
        self.activation = nn.ReLU()
        self.loss = nn.MSELoss()

        encoder = nn.Sequential(
            nn.Linear(28 * 28, 64), 
            self.activation, 
            nn.Linear(64, 3))
        decoder = nn.Sequential(
            nn.Linear(3, 64), 
            self.activation,
            nn.Linear(64, 28 * 28))
        
        self.encoder = encoder
        self.decoder = decoder

        # Initialize weights. This is not necessary for small networks, but good practise
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        # log the hyperparameters to the checkpoints
        self.save_hyperparameters()


    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # training_step defines the train loop it is independent of forward
        # deconstruct input and output from training data
        # for the autoencoder, we don't need labels
        x, y = batch

        # this calls forward()
        x_hat = self(x)

        # compute loss against input (we want to learn an encoding)
        loss = self.loss(x_hat, x)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch:Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        # This configures the optimizer just as in the first example.
        # We also employ an adaptive learning rate scheduler.
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
