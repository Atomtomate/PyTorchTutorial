import sys
sys.dont_write_bytecode = True

import torch
from os.path import dirname, abspath, join

import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler
import json
from models import ConvEncoder, ConvDecoder, CVAE2
from data import CVAE_MNIST_Data

# set dtype, float64 is almost never necessary
dtype_default = torch.float32
torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(dtype_default)
device = "cuda" if torch.cuda.is_available() else "cpu"



from argparse import ArgumentParser


def main(args):
    config = json.load(open(join(dirname(abspath(__file__)),'config.json')))
    torch.manual_seed(config['seed'])
    dataMod = CVAE_MNIST_Data(config)
    model = CVAE2(config, device, mode = 'linear') 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("lightning_logs", name="VAE_Linear")
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.8f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last =True
            )
    early_stopping = EarlyStopping(monitor="val_loss",
                                patience=40)
    swa = StochasticWeightAveraging(swa_lrs=1e-8,
                                    annealing_epochs=40,
                                    swa_epoch_start=220,
                                    )
    accumulator = GradientAccumulationScheduler(scheduling={0: 128, 12: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
    trainer = Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                      callbacks=callbacks, logger=logger, gradient_clip_val=0.5) #precision="16-mixed", 



    trainer.fit(model, datamodule=dataMod)


if __name__ == "__main__":
    # This is how you would parse arguments from the command line,
    # as you would need to, for example in slurm scripts.

    parser = ArgumentParser()
    # parser.add_argument("--accelerator", default=None)
    # parser.add_argument("--devices", default=None)
    args = parser.parse_args()
    main(args)
