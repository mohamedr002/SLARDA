
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.reduced_cnn_size = 128
        self.num_classes = 5
        self.dropout = 0.5

        self.wide_kernel_size = 400
        self.wide_stride_size = 50

        self.small_kernel_size = 50
        self.small_stride_size = 6

        # Reset Configs
        self.num_filters = 16
        self.stride = 8
        self.kernel_size = 50

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

