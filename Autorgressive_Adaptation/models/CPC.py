import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
from args import args

device = args.device

class CPC(nn.Module):
    def __init__(self, num_channels, gru_hidden_dim, timestep):
        super(CPC, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = gru_hidden_dim
        self.gru = nn.GRU(num_channels, self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.timestep = timestep
        self.Wk = nn.ModuleList([nn.Linear(self.hidden_dim, num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.logits = nn.Linear(gru_hidden_dim, 5)

        # self.seq_transformer = Seq_Transformer(patch_size=30, dim=gru_hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug):
        z_aug = features_aug  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug.shape[2]
        z_aug = z_aug.transpose(1, 2)



        batch = z_aug.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(device)  # e.g. size 12*8*512

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug[:, t_samples + i, :].view(batch, self.num_channels)  # z_tk e.g. size 8*512
        forward_seq = z_aug[:, :t_samples + 1, :]  # e.g. size 8*100*512
        output, _ = self.gru(forward_seq)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, self.hidden_dim)  # c_t e.g. size 8*256
        # c_t = self.seq_transformer(forward_seq)
        # c_t = output[:, t_samples, :].view(batch, self.hidden_dim * 2)  # c_t e.g. size 8*256
        # c_t = c_t[:, :self.hidden_dim] + c_t[:,self.hidden_dim:]
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(device) # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        return nce, c_t