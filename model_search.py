import pytorch_lightning as pl
import random

from model import TinySleepNet
from lightning_wrapper import LightningWrapper
from benchmark import benchmark

random.seed(42)
results = []
for conv1_ch in [128, 96, 64, 32]:
    for conv1_kern in [50, 40, 32, 16]:
        model = LightningWrapper(TinySleepNet(conv1_ch=conv1_ch, conv1_kern=conv1_kern))
        trainer = pl.Trainer(reload_dataloaders_every_epoch=True, gpus=1, max_epochs=450)
        trainer.fit(model)
        print(f"{conv1_ch} {conv1_kern} {model.max_acc}")
        bench = benchmark(model.net, num_tests=1000)
        print(f"bench {bench}")
        results.append((conv1_ch, conv1_kern, model.max_acc, bench))

print('Final results:')
for entry in results:
    print(f'{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}')
