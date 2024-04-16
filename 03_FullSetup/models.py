# Model taken from this Tutorial:
# https://www.kaggle.com/code/kushal1506/pytorch-vae-mnist
# https://www.tensorflow.org/tutorials/generative/cvae
# https://skannai.medium.com/what-are-convolutional-variational-auto-encoders-cvae-515f4fedc23
# Feature extractor code: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torch.nn import functional as F
import torchvision

from typing import Dict, Iterable, Callable


def activation_str(act_str: str):
    if act_str.lower() == "relu":
        return nn.ReLU()
    elif act_str.lower() == "silu":
        return nn.SiLU()
    elif act_str.lower() == "leakyrelu":
        return nn.LeakyReLU()


class ConvEncoder(nn.Module):
    """
    Encoder Architecture taken from here: https://www.kaggle.com/code/kushal1506/pytorch-vae-mnist
    """
    def __init__(self, latent_dim, activation, kernel_size, stride, z_dim):
        super(ConvEncoder,self).__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride= stride
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,kernel_size,padding=1),
            nn.BatchNorm2d(32),
            activation,
            nn.Conv2d(32,64,kernel_size,padding=1,stride=stride),
            nn.BatchNorm2d(64),
            activation,
            nn.Conv2d(64,64,kernel_size,padding=1,stride=stride),
            nn.BatchNorm2d(64),
            activation,
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)
        
        
class ConvDecoder(nn.Module):
    """
    Decoder Architecture taken from here: https://www.kaggle.com/code/kushal1506/pytorch-vae-mnist
    """
    def __init__(self, latent_dim, activation, kernel_size, stride, z_dim): 
        super(ConvDecoder,self).__init__()  
        self.activation = activation
        self.kernel_size = kernel_size
        self.stride= stride
        self.z_dim = z_dim    
        #TODO: hardcoded from default config, compute this here
        hidden_size = 64*7*7
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.Unflatten(1,(64,7,7)),
            activation,
            nn.ConvTranspose2d(64,64,kernel_size,padding=1,output_padding=1,stride=stride),
            nn.BatchNorm2d(64),
            activation,
            nn.ConvTranspose2d(64,32,kernel_size,padding=1,output_padding=1,stride=stride),
            nn.BatchNorm2d(32),
            activation,
            nn.ConvTranspose2d(32, 1, kernel_size, padding=1),
            )

    def forward(self, x):
        return self.decoder(x)
    
class LatentZ(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        #TODO: hardcoded from default config, compute this here
        self.hidden_size = hidden_size 
        self.latent_size = latent_size
        self.mu = nn.Linear(self.hidden_size, latent_size)
        self.logvar = nn.Linear(self.hidden_size, latent_size)

    def forward(self, p_x):
        mu = self.mu(p_x)
        logvar = self.logvar(p_x)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, mu, logvar

    
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, activation=nn.LeakyReLU(0.2)):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, latent_dim)
            )
    def forward(self, x):
        return self.encoder(x.view(-1, 784))

class Decoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, activation=nn.LeakyReLU(0.2)):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.decoder(x)


class CVAE2(L.LightningModule):
    """
    Model taken from this Tutorial: 
    https://bytepawn.com/building-a-pytorch-autoencoder-for-mnist-digits.html and
    https://www.tensorflow.org/tutorials/generative/cvae
    https://skannai.medium.com/what-are-convolutional-variational-auto-encoders-cvae-515f4fedc23
    """
    def __init__(self, config, device, mode='linear'):
        super(CVAE2, self).__init__()

        self.activation = activation_str(config['AE_activation'])
        self.latent_dim = config['latent_dim']
        self.c_device = device
        self.sigmoid = nn.Sigmoid()
        self.log_likelihood = nn.BCEWithLogitsLoss(reduction='sum')
        #nn.MSELoss()
        self.mode= mode

        #TODO self.save_hyperparams
        
        if mode == 'conv':
            self.z_dim=config['conv_AE_z_dim']
            self.encoder = ConvEncoder(self.latent_dim, 
                                       activation=self.activation,
                                       kernel_size=config['conv_AE_kernel_size'],
                                       stride=config['conv_AE_stride'],
                                       z_dim=self.z_dim) 
            self.decoder = ConvDecoder(self.latent_dim, 
                                       activation=self.activation,
                                       kernel_size=config['conv_AE_kernel_size'],
                                       stride=config['conv_AE_stride'],
                                       z_dim=self.z_dim)
            #TODO: hidden size hardcoded from default config, compute this here
            self.latent_z = LatentZ(64*7*7, self.z_dim)#
        elif mode == 'linear':
            self.encoder = Encoder(input_dim=config['linear_AE_input_dim'],
                                   hidden_dim=config['linear_AE_hidden_dim'], 
                                   latent_dim=config['linear_AE_latent_dim'], 
                                   activation=self.activation
                                   ) 
            self.decoder = Decoder(input_dim=config['linear_AE_input_dim'],
                                   hidden_dim=config['linear_AE_hidden_dim'], 
                                   latent_dim=config['linear_AE_latent_dim'], 
                                   activation=self.activation
                                   ) 
            self.z_dim = config['linear_AE_latent_dim']
            self.latent_z = LatentZ(config['linear_AE_latent_dim'], self.z_dim)#
            
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        self.lr = config['lr']
        self.save_hyperparameters()

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def encode(self, x):
        x = self.encoder(x)
        return x
    

    def forward(self, x):
        p_x = self.encode(x)
        z, mu, logvar = self.latent_z(p_x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def loss_function(self, x, x_hat, mean, logvar):
        # Reproduction loss of image
        
        reproduction_loss = self.log_likelihood(x, x_hat)
        #reproduction_loss = self.log_likelihood(x, x_hat)
        # Kullback-Leibler divergence loss to maximize the evidence lower bound on the marginal log-likelihood
        # https://mbernste.github.io/posts/elbo/
        KLD = - 0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reproduction_loss, KLD
    
    def criterion(self, logits, targets):
        return F.cross_entropy(logits, targets)

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs, mean, logvar = self(inputs)
        reproduction_loss, KLD = self.loss_function(inputs.view(-1, 784), outputs, mean, logvar)
        loss = reproduction_loss + KLD
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_reproduction_loss", reproduction_loss, prog_bar=False)
        self.log("train_KLD", KLD, prog_bar=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs, mean, logvar = self(inputs)
        reproduction_loss, KLD = self.loss_function(inputs.view(-1, 784), outputs, mean, logvar)
        loss = reproduction_loss + KLD
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_reproduction_loss", reproduction_loss, prog_bar=False)
        self.log("val_KLD", KLD, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, mean, logvar = self(inputs)
        reproduction_loss, KLD = self.loss_function(inputs, outputs, mean, logvar)
        loss = reproduction_loss + KLD
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_reproduction_loss", reproduction_loss, prog_bar=False)
        self.log("test_KLD", KLD, prog_bar=False)

        # log 6 example images
        # or generated text... or whatever
        if batch_idx == 0:
            sample_imgs = x[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        labels_hat = torch.argmax(out, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs!
        self.log_dict({'test_loss': loss, 
                       'test_acc': test_acc, 
                       'test_reproduction_loss': reproduction_loss,
                       'test_KLD': KLD})

    
class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        This is intentionally a different style than the CVAE above, where the network was wrapped in a sequantial!
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        # NLL, or Softmax for multi class classification
        # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
        loss = F.nll_loss(targets, outputs)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self(inputs)
        loss = F.nll_loss(targets, outputs)
        self.log("train_loss", loss, prog_bar=False)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)
        return optimizer
    

class FeatureExtractor(nn.Module):
    """
    From https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    """
    def __init__(self, model: nn.Module, layers: Dict):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers.values()}

        for layer in layers:
            layer = dict([*self.model.named_modules()])[layer]
            layer.register_forward_hook(self.save_outputs_hook(layer))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features