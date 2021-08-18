
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 3
        self.out_channels = 32
        self.kernel_size = 8
        self.feat_dim = 32

        self.dropout = 0.5
        self.num_classes = 6

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
        self.apply_transform = True
        self.class_names = ['Biking', 'Sitting', 'Standing', 'Walking', 'Stair Up', 'Stair down']

        self.SimCLR = SimCLR_configs()


class SimCLR_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True

