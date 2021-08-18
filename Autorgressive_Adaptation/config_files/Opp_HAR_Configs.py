class Config(object):
    def __init__(self):
        # model configs
        self.out_channels = 16
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1
        self.input_channels = 3
        self.kernel_size = 8
        self.feat_dim = 16

        # Resnet Configs
        self.num_filters = 8
        self.stride = 1
        # training configs
        self.num_epoch = 15
        self.batch_size = 256

        # Att Disc Parameters
        self.att_hid_dim = 64
        self.patch_size = self.out_channels
        self.depth =8
        self.heads = 2
        self.mlp_dim = 64


        # data parameters
        self.shuffle = True
        self.drop_last = True
        self.apply_transform = True
        self.class_names = ['sit', 'stand', 'lie', 'walk']
        self.num_classes=4
        self.base_model = base_model_configs()

        self.SLARDA= SLARDA_Configs()

class base_model_configs(object):
    def __init__(self):
        # model configs
        self.input_channels = 113
        self.out_channels = 16
        self.kernel_size = 8
        self.feat_dim = 16

        self.dropout = 0.5
        self.num_classes = 4

class SLARDA_Configs(object):
    def __init__(self):
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 0.5e-4
        self.lr_disc = 0.5e-4
        self.AR= 'ATT'
        self.gamma = 0.1
        self.step_size = 5

        self.teacher_wt = 0.1
        self.confidence_level = 0.9
        self.momentum_wt = 0.996


class Supervised(object):
    def __init__(self):

        # training configs
        self.num_epoch = 30
        self.save_ckp = True
        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 3e-4
