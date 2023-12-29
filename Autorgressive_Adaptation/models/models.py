import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from abc import abstractmethod
import math
import sys
sys.path.append("..")
from args import args
from models.attention import Seq_Transformer
class CNN_Opp_HAR_SL(nn.Module):
    def __init__(self, configs):
        super(CNN_Opp_HAR_SL, self).__init__()
        self.input_dim = configs.input_channels
        self.out_channels = configs.out_channels
        self.hidden_dim = configs.feat_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(configs.input_channels, 16, kernel_size=configs.kernel_size, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Conv1d(16, 16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, configs.out_channels, kernel_size=3, stride=2),
            nn.BatchNorm1d(configs.out_channels),
            nn.ReLU())
        self.flatten = nn.AdaptiveAvgPool1d(1)
        self.Classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, configs.num_classes))
        self.cpc = CPC(configs.out_channels, 16, 12)
    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        # src = src.view(src.size(0), self.input_dim, -1)
        seq_features = self.encoder(src).squeeze()
        vec_features = self.flatten(seq_features).squeeze()
        predictions = self.Classifier(vec_features)
        return predictions, (vec_features,seq_features)
class CNN_BN(nn.Module):
    def __init__(self, configs):
        super(CNN_BN, self).__init__()
        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.num_classes
        self.AR = configs.AR
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

    def forward(self, src):
        # reshape input (batch_size, input_dim, sequence length)
        src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        features = self.med_layer(full_features)
        predictions = self.Classifier(features)

        return predictions, (features,full_features)
class CNN_SL_bn(nn.Module):
    def __init__(self, configs):
        super(CNN_SL_bn, self).__init__()

        self.input_dim = configs.input_channels
        self.dropout = configs.dropout
        self.kernel_size = configs.kernel_size
        self.hidden_dim = configs.cls_hidden_dim
        self.out_dim = configs.num_classes
        self.AR = configs.AR
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
        # src = src.view(src.size(0), self.input_dim, -1)
        full_features = self.encoder(src)
        features = self.med_layer(full_features)
        predictions = self.Classifier(features)
        return predictions, (features,full_features)

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
        features,_ = self.med_layer(full_features)
        predictions = self.Classifier(features[:,-1,:])
        return predictions, (features[:,-1,:],full_features)


class CPC(nn.Module):
    def __init__(self, num_channels, gru_hidden_dim, timestep):
        super(CPC, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = gru_hidden_dim
        self.gru = nn.GRU(num_channels, self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.timestep = timestep
        self.Wk = nn.ModuleList([nn.Linear(self.hidden_dim, num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, features):
        z = features  # features are (batch_size, #channels, seq_len)
        # seq_len = z.shape[2]
        z = z.transpose(1, 2)


        batch = z.shape[0]
        t_samples = torch.randint(self.timestep, size=(1,)).long()  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float()  # e.g. size 12*8*512

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.num_channels)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, _ = self.gru(forward_seq)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.hidden_dim)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.num_channels)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        return nce
class EEG_M(nn.Module):
    def __init__(self, configs):
        super(EEG_M, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size, stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
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
        self.cpc = CPC(configs.reduced_cnn_size, 64, 30)

    def forward(self, x_in):
        full_features = self.feature_extractor(x_in)
        vec_features = full_features.view(full_features.shape[0], -1)
        logits = self.logits(vec_features)
        return logits, (vec_features, full_features)
class EEG_M_SL(nn.Module):
    def __init__(self, configs):
        super(EEG_M_SL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size, stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
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
        vec_features = full_features.view(full_features.shape[0], -1)
        logits = self.logits(vec_features)
        return logits, (vec_features, full_features)
    def get_parameters(self):
        parameter_list = [{"params": self.feature_extractor.parameters(), "lr_mult": 0.01, 'decay_mult': 2},
                          {"params": self.logits.parameters(), "lr_mult": 0.01, 'decay_mult': 2}, ]
        return parameter_list
class Discriminator_AR(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        self.input_dim = configs.out_channels
        super(Discriminator_AR, self).__init__()

        self.AR_disc = nn.GRU(input_size=self.input_dim, hidden_size=configs.disc_AR_hid,num_layers = configs.disc_n_layers,bidirectional=configs.disc_AR_bid, batch_first=True)
        self.DC = nn.Linear(configs.disc_AR_hid+configs.disc_AR_hid*configs.disc_AR_bid, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.input_dim )
        encoder_outputs, (encoder_hidden) = self.AR_disc(input)
        features = encoder_outputs[:, -1, :]
        domain_output = self.DC(features)
        return domain_output
    def get_parameters(self):
        parameter_list = [{"params":self.AR_disc.parameters(), "lr_mult":0.01, 'decay_mult':1}, {"params":self.DC.parameters(), "lr_mult":0.01, 'decay_mult':1},]
        return parameter_list
class Discriminator_ATT(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        self.patch_size =  configs.patch_size
        self.hid_dim = configs.att_hid_dim
        self.depth= configs.depth
        self.heads = configs.heads
        self.mlp_dim = configs.mlp_dim
        super(Discriminator_ATT, self).__init__()
        self.transformer= Seq_Transformer(patch_size=self.patch_size, dim=configs.att_hid_dim, depth=self.depth, heads= self.heads , mlp_dim=self.mlp_dim)
        self.DC = nn.Linear(configs.att_hid_dim, 1)
    def forward(self, input):
        """Forward the discriminator."""
        # src_shape = [batch_size, seq_len, input_dim]
        input = input.view(input.size(0),-1, self.patch_size )
        features = self.transformer(input)
        domain_output = self.DC(features)
        return domain_output
class Discriminator(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.feat_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1)
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
