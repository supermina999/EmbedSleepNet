import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import random

from model import TinySleepNet, EmbedSleepNet
from lightning_wrapper import LightningWrapper

random.seed(42)

model = LightningWrapper(EmbedSleepNet())

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(logger=tb_logger, reload_dataloaders_every_epoch=True, gpus=1)
trainer.fit(model)
