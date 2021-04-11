import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class SleepDataset(Dataset):
    def __init__(self, data, labels, minibatch_size=20):
        total_len = 0
        for subj_data in data:
            total_len += subj_data.shape[0] // minibatch_size

        self.data = data
        self.labels = labels
        self.minibatch_size = minibatch_size
        self.total_len = total_len
        self.reshuffle()

    @classmethod
    def load_data(cls, data_path='preprocessed'):
        data = []
        labels = []
        for file in os.listdir(data_path):
            if not file.endswith('.npz'):
                continue
            path = os.path.join(data_path, file)
            loaded = np.load(path)
            data.append(loaded['data'])
            labels.append(loaded['labels'])
        return data, labels

    def reshuffle(self):
        self.shifts = []
        for subj_data in self.data:
            self.shifts.append(random.randint(0, subj_data.shape[0] % self.minibatch_size))

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.data) - 1
        for idx, subj_data in enumerate(self.data):
            cur_len = subj_data.shape[0] // self.minibatch_size
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        start_idx = self.shifts[subj_idx] + index * self.minibatch_size
        item_data = self.data[subj_idx][start_idx:start_idx+self.minibatch_size]
        item_labels = self.labels[subj_idx][start_idx:start_idx+self.minibatch_size]
        return torch.tensor(item_data, dtype=torch.float), torch.tensor(item_labels, dtype=torch.long)


def load_split_sleep_dataset(data_path='preprocessed', minibatch_size=20, train_coef=0.8):
    data, labels = SleepDataset.load_data(data_path)
    train_size = round(len(data) * train_coef)
    data_train, labels_train = data[:train_size], labels[:train_size]
    data_test, labels_test = data[train_size:], labels[train_size:]
    return SleepDataset(data_train, labels_train, minibatch_size), SleepDataset(data_test, labels_test, minibatch_size)
