import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from model import LightningModule

model = LightningModule()

tb_logger = pl_loggers.TensorBoardLogger('logs/')
trainer = pl.Trainer(logger=tb_logger, reload_dataloaders_every_epoch=True, gpus=1, auto_lr_find=True)
trainer.tune(model)
trainer.fit(model)
