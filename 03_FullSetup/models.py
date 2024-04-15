# Model taken from this Tutorial:
# https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html and
# https://www.tensorflow.org/tutorials/generative/cvae
# https://skannai.medium.com/what-are-convolutional-variational-auto-encoders-cvae-515f4fedc23

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L


def activation_str(act_str: str):
    if act_str.lower() == "relu":
        return nn.ReLU()
    elif act_str.lower() == "silu":
        return nn.SiLU()
    elif act_str.lower() == "leakyrelu":
        return nn.LeakyReLU()


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, activation):
        super(ConvEncoder,self).__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            activation,
            nn.Conv2d(4, 8, kernel_size=5),
            activation,
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, latent_dim)
        )

    def forward(self, x):
        self.encoder(x)
        
        
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, activation): 
        super(ConvDecoder,self).__init__()      
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            # 400
            activation,
            nn.Linear(400, 4000),
            # 4000
            activation,
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            )
        
    def forward(self, x):
        self.decoder(x)

class CVAE2(L.LightningModule):
    """
    Model taken from this Tutorial: 
    https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html and
    https://www.tensorflow.org/tutorials/generative/cvae
    https://skannai.medium.com/what-are-convolutional-variational-auto-encoders-cvae-515f4fedc23
    """
    def __init__(self, config):
        super().__init__()

        self.activation = activation_str(config['AE_activation'])
        self.latent_dim = config['latent_dim']
        self.lossF = nn.MSELoss()
        
        #TODO self.save_hyperparams

        self.encoder = ConvEncoder(self.latent_dim, self.activation)
        self.decoder = ConvDecoder(self.latent_dim, self.activation)
        self.mean_layer = nn.Linear(self.latent_dim, 2)
        self.logvar_layer = nn.Linear(self.latent_dim, 2)
        
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        y = self.reparameterization(mean, logvar)
        x_hat = self.decoder(y)
        return x_hat
        
    # define the loss function
    def criterion(self, logits, targets):
        return F.cross_entropy(logits, targets)

    # process inside the training loop
    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        #loss = self.criterion(outputs, targets)
        loss = self.lossF(inputs, outputs)
        #inbuilt tensorboard for logs

        self.log("train_loss", loss, prog_bar=False)
        return loss


    # process inside the validation loop
    def validation_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = self.lossF(inputs, outputs)

        #loss = self.criterion(outputs, targets)

        # Accuracy calculation
        #pred = outputs.data.max(1)[1]  # get the index of the max log-probability
        #incorrect = pred.ne(targets.long().data).cpu().sum()
        #err = incorrect.item()/targets.numel()
        #val_acc = torch.tensor(1.0-err)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    #return average loss and accuracy after every epoch
    #def validation_epoch_end(self, outputs):
    #    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #    avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()  
    #    Accuracy = 100 * avg_acc.item()
    #    tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_acc}
    #    print('Val Loss:', round(avg_loss.item(),2), 'Val Accuracy: %f %%' % Accuracy) 
    #    return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}


    # Can return multiple optimizers and scheduling alogoithms 
    # Here using Stuochastic Gradient Descent
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        return optimizer