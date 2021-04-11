import torch.nn.functional as F
import torch.optim
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import load_split_sleep_dataset


class TinySleepNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 50, 6)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.maxpool1 = nn.MaxPool1d(8, 8)
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(128, 8, 1)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(8, 8, 1)
        self.batchnorm3 = nn.BatchNorm1d(8)
        self.conv4 = nn.Conv1d(8, 8, 1)
        self.batchnorm4 = nn.BatchNorm1d(8)
        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        old_shape = x.shape
        x = x.reshape((x.shape[0] * x.shape[1], 1, -1))
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout1(self.maxpool1(x))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = x.reshape(old_shape[0], old_shape[1], -1)

        return x

class TinySleepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = TinySleepNetCNN()
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc(x).squeeze(1)
        return x


class LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TinySleepNet()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = 1e-3

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        outputs = self(data)
        outputs = outputs.reshape(-1, outputs.shape[2])
        labels = labels.reshape(-1)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        outputs = self(data)
        outputs = outputs.reshape(-1, outputs.shape[len(outputs.shape) - 1])
        labels = labels.reshape(-1)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss.item(), prog_bar=True)
        return {'val_loss': loss.item()}

    def prepare_data(self):
        self.train_ds, self.val_ds = load_split_sleep_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=3)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=64)

    def on_train_epoch_start(self):
        self.train_ds.reshuffle()
        self.val_ds.reshuffle()
