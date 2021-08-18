from torch import nn
from torch.autograd import Function
from .CPC import CPC
from args import  args
training_mode = args.training_mode

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()
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
        full_features = self.encoder(src).squeeze()
        if training_mode == "self_supervised":
            return self.cpc(full_features)
        else:
            features = self.flatten(full_features)
            predictions = self.Classifier(features.squeeze())
            return predictions, features
        # return predictions, (vec_features,seq_features)

class Discriminator(nn.Module):
    """Discriminator model for source domain."""
    def __init__(self, input_dims=128, hidden_dims=20, output_dims=1):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input.squeeze())
        return out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None