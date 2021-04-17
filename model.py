import torch.nn.functional as F
import torch.optim
from torch import nn
import math


class TinySleepNetCNN(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv1_ch, conv1_kern, 6)
        self.batchnorm1 = nn.BatchNorm1d(conv1_ch)
        self.maxpool1 = nn.MaxPool1d(8, 8)
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv1d(conv1_ch, 8, 1)
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
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.dropout1(self.maxpool1(x))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)

        return x


class TinySleepNetCNN2(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv1_ch, conv1_kern, 6)
        self.maxpool1 = nn.MaxPool1d(8, 8)
        self.conv2 = nn.Conv1d(conv1_ch, 8, 1)
        self.conv3 = nn.Conv1d(8, 8, 1)
        self.conv4 = nn.Conv1d(8, 8, 1)
        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        old_shape = x.shape
        x = x.chunk(x.shape[0], 0)
        x = torch.cat(x, 1).squeeze(0)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.maxpool1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = x.unsqueeze(0).chunk(old_shape[0], 1)
        x = torch.cat(x)

        return x


class TinySleepNet(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.cnn = TinySleepNetCNN(conv1_ch=conv1_ch, conv1_kern=conv1_kern)
        self.lstm = nn.LSTM(input_size=120, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ninp, heads, hidden_size, layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src_mask = self.generate_square_subsequent_mask(src.shape[0]).to(src.device)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


class MySleepNet(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.cnn = TinySleepNetCNN(conv1_ch=conv1_ch, conv1_kern=conv1_kern)
        self.fc1 = nn.Linear(120, 32)
        self.fc2 = nn.Linear(32, 5)
        self.trf = TransformerModel(32, 1, 64, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        x = x.permute(1, 0, 2)
        x = self.trf(x)
        x = x.permute(1, 0, 2)
        x = self.fc2(x)
        return x


class MySleepNet2(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.cnn = TinySleepNetCNN(conv1_ch=conv1_ch, conv1_kern=conv1_kern)
        self.fc1 = nn.Linear(120, 16)
        self.seq = nn.Sequential(
            nn.Conv2d(1, 2, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 8, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 1, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        x = self.seq(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return x


class MySleepNet22(nn.Module):
    def __init__(self, conv1_ch=128, conv1_kern=50):
        super().__init__()
        self.cnn = TinySleepNetCNN2(conv1_ch=conv1_ch, conv1_kern=conv1_kern)
        self.fc1 = nn.Linear(120, 16)
        self.seq = nn.Sequential(
            nn.Conv2d(1, 2, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 8, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 4, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4, 1, (5, 5), 1, (2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        x = self.seq(x.unsqueeze(1)).squeeze(1)
        x = self.fc2(x)
        return x


class MySleepNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(1, 4, 19, 5, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(4, 8, 19, 5, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 16, 19, 5, 9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 11, 3, 5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, 7, 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, 7, 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 7, 2, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(64, 5)

    def forward(self, x):
        x = self.seq(x.reshape(x.shape[0], 1, -1)).permute(0, 2, 1)
        x = self.fc(x)
        return x
