import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, data_path='preprocessed'):
        data = []
        labels = []
        for file in os.listdir(data_path):
            if not file.endswith('.npz'):
                continue
            path = os.path.join(data_path, file)
            loaded = np.load(path)
            data.append(loaded['data'])
            labels.append(loaded['labels'])
        self.data = np.concatenate(data)
        self.labels = np.concatenate(labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float), self.labels[index]
