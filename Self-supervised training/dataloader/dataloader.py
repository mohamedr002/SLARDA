import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np
from .augmentations import DataTransform

import sys

sys.path.append("..")
from args import args

training_mode = args.training_mode
use_SimCLR = args.use_SimCLR  # to check if augmentations are needed or not!


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, apply_transform):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2) #X_train.unsqueeze(2)   # np.expand_dims(X_train, 2)
        self.num_channels = min(X_train.shape)
        if X_train.shape.index(self.num_channels) != 1:  # data dim is #samples, seq_len, #channels
            X_train = X_train.permute(0, 2, 1)
            # X_train = X_train.transpose(0,2,1)

        if apply_transform:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None
        self.len = X_train.shape[0]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, domain, configs):

    train_dataset = torch.load(os.path.join(data_path, f"train_{domain}.pt"))
    valid_dataset = torch.load(os.path.join(data_path, f"test_{domain}.pt"))
    test_dataset = torch.load(os.path.join(data_path, f"test_{domain}.pt"))

    train_dataset = Load_Dataset(train_dataset, False)
    valid_dataset = Load_Dataset(valid_dataset, False)
    test_dataset = Load_Dataset(test_dataset, False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
