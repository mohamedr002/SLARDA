import sys
sys.path.append("..")
from torch import nn
from .CPC import CPC
from args import args

training_mode = args.training_mode

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.num_classes
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=self.kernel_size, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1240, self.hidden_dim))
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim))
        self.cpc = CPC(8, 64, 30)

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        if training_mode == "self_supervised":
            return self.cpc(full_features)
        else:
            features = self.med_layer(full_features)
            predictions = self.Classifier(features)
            return predictions, features


class CNN_AR(nn.Module):
    def __init__(self, configs):
        super(CNN_AR, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.num_classes
        self.feature_dim = configs.cnn_feat_dim
        self.training_mode  = configs.training_mode
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=self.kernel_size, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1))
        self.med_layer = nn.GRU(self.feature_dim, self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim))
        self.cpc = CPC(8, 64, 30)

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        if self.training_mode == "self_supervised":
            return self.cpc(full_features)
        else:
            features,_ = self.med_layer(full_features)
            predictions = self.Classifier(features[:,-1,:])
            return predictions, features

class base_model_BN(nn.Module):
    def __init__(self, configs):
        super(base_model_BN, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.num_classes
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=self.kernel_size, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1240, self.hidden_dim))
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim))
        self.cpc = CPC(8, 64, 30)

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        if training_mode == "self_supervised":
            return self.cpc(full_features)
        else:
            features = self.med_layer(full_features)
            predictions = self.Classifier(features)
            return predictions, features

class CNN_CPC(nn.Module):
    def __init__(self, configs):
        super(CNN_CPC, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.cls_hidden_dim
        self.AR = configs.AR
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=self.kernel_size, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1240, self.hidden_dim))
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim))
        self.cpc = CPC(8, 64, 30)

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        features = self.med_layer(full_features)
        predictions = self.Classifier(features)
        features = full_features
        return predictions, (features,full_features)


class CNN_CPC_GRU(nn.Module):
    def __init__(self, configs):
        super(CNN_CPC_GRU, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.cls_hidden_dim
        self.AR = configs.AR
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 8, kernel_size=self.kernel_size, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=8, stride=1, padding=1, dilation=1))
        self.med_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1240, self.hidden_dim))
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.out_dim))
        self.cpc = CPC(8, 64, 30)

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        features = self.med_layer(full_features)
        predictions = self.Classifier(features)
        features = full_features
        return predictions, (features,full_features)
