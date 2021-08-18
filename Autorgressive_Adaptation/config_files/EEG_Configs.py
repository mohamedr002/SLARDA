
class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.reduced_cnn_size = 128
        self.out_channels = 128
        self.feat_dim = 10240
        self.num_classes = 5

        # Resnet Configs
        self.num_filters = 16
        self.stride = 8
        self.kernel_size = 50


        # GRU configs
        self.disc_n_layers = 1
        self.disc_AR_hid = 512
        self.disc_AR_bid = False
        self.disc_hid_dim = 100
        self.disc_out_dim = 1

        # Att Disc Parameters
        self.att_hid_dim = 512
        self.patch_size = self.out_channels
        self.depth = 8
        self.heads = 4
        self.mlp_dim = 64

        # training hyper-param
        self.num_epoch = 15
        self.batch_size = 512
        self.save_ckp = True

        # data parameters
        self.shuffle = True
        self.drop_last = True
        self.apply_transform = True
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.base_model = base_model_configs()
        self.Supervised = Supervised_configs()
        self.SLARDA = SLARDA_Configs()
        # self.CPC_DA = CPC_DA_Configs

class Supervised_configs(object):
    def __init__(self):
        # training configs
        self.num_epoch = 25
        self.batch_size = 128
        self.save_ckp = True
        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 3e-4

class base_model_configs(object):
    def __init__(self):
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

# Cross-domain Configs
class SLARDA_Configs(object):
    def __init__(self):
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 1e-3
        self.lr_disc = 1e-3
        self.AR= 'GRU'
        self.gamma = 0.1
        self.step_size = 10

        self.teacher_wt =0.1
        self.confidence_level = 0.9
        self.momentum_wt = 0.996