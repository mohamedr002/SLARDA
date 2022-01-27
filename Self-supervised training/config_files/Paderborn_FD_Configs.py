
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.reduced_cnn_size = 8
        self.num_classes = 3
        self.dropout = 0.5
        self.training_mode = 'self_supervised'
        self.cnn_feat_dim = 155

        self.kernel_size = 32
        self.cls_hidden_dim = 32

        # training configs
        self.num_epoch = 30
        self.batch_size = 128

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']

        self.SimCLR = SimCLR_configs()


class SimCLR_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True

