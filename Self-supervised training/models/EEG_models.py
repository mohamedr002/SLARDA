import torch
from torch import nn

from .CPC import CPC
import sys

sys.path.append("..")
from args import args

training_mode = args.training_mode
use_SimCLR = args.use_SimCLR


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=configs.small_kernel_size,
                      stride=configs.small_stride_size, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(configs.dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=configs.wide_kernel_size, stride=configs.wide_stride_size,
                      bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(configs.dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, configs.reduced_cnn_size, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(configs.reduced_cnn_size),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(configs.dropout)

        model_output_dim = 80
        self.logits = nn.Linear(model_output_dim * configs.reduced_cnn_size, configs.num_classes)

        self.cpc = CPC(configs.reduced_cnn_size, 64, 30)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)

        if training_mode == "self_supervised" and use_SimCLR is False:
            return self.cpc(x_concat)
        else:
            x_concat = x_concat.view(x_concat.shape[0], -1)
            logits = self.logits(x_concat)
            return logits, x_concat


class CNN_EEG_SL(nn.Module):
    def __init__(self, configs):
        super(CNN_EEG_SL, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=configs.small_kernel_size,
                      stride=configs.small_stride_size, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(configs.dropout),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=configs.wide_kernel_size, stride=configs.wide_stride_size,
                      bias=False, padding=200),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(configs.dropout),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, configs.reduced_cnn_size, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(configs.reduced_cnn_size),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(configs.dropout)

        model_output_dim = 80
        self.logits = nn.Linear(model_output_dim * configs.reduced_cnn_size, configs.num_classes)

        self.cpc = CPC(configs.reduced_cnn_size, 64, 30)
        self.aap = nn.AdaptiveAvgPool1d(model_output_dim)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        full_features = self.dropout(x_concat)
        full_features = self.aap(full_features)
        if training_mode == "self_supervised" and use_SimCLR is False:
            return self.cpc(full_features)
        else:
            vec_features = full_features.view(full_features.shape[0], -1)
            logits = self.logits(vec_features)
            return logits, (vec_features, full_features)


class EEG_M(nn.Module):
    def __init__(self, configs):
        super(EEG_M, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size, stride=configs.stride, bias=False,
                      padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout),
            nn.Conv1d(64, configs.reduced_cnn_size, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.reduced_cnn_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(configs.dropout)
        )
        model_output_dim = 22
        self.logits = nn.Linear(model_output_dim * configs.reduced_cnn_size, configs.num_classes)
        self.cpc = CPC(configs.reduced_cnn_size, 64, 10)

    def forward(self, x_in):
        full_features = self.feature_extractor(x_in)
        if training_mode == "self_supervised" and use_SimCLR is False:
            return self.cpc(full_features)
        else:
            vec_features = full_features.view(full_features.shape[0], -1)
            logits = self.logits(vec_features)
            return logits, vec_features
