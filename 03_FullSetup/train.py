import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler
import lightning as L
import json
from models import *

# set dtype, float64 is almost never necessary
dtype_default = torch.float32
torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(dtype_default)


config = json.load(open('config.json'))
torch.manual_seed(config['seed'])
model = CVAE(config) 
lr_monitor = LearningRateMonitor(logging_interval='step')
early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=40)
logger = TensorBoardLogger("lightning_logs", name=config["model_name"])
val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
        filename="{epoch}-{step}-{val_loss:.8f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last =True
        )
swa = StochasticWeightAveraging(swa_lrs=1e-8,
                                annealing_epochs=40,
                                swa_epoch_start=220,
                                )
#accumulator = GradientAccumulationScheduler(scheduling={0: 128, 8: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
callbacks = [lr_monitor, early_stopping] #val_ckeckpoint,  DeviceStatsMonitor(), swa, accumulator
trainer = L.Trainer(max_epochs=config["epochs"], accelerator="gpu")#, callbacks=callbacks, logger=logger) #precision="16-mixed", 


#tuner = Tuner(trainer)
#lr_find_results = tuner.lr_find(model, min_lr=1e-04, num_training=300)
#fig = lr_find_results.plot(suggest=True)
#logger.experiment.add_figure("lr_find", fig)
#new_lr = lr_find_results.suggestion()
#model.hparams.lr = new_lr

trainer.fit(model)