class Config(object):
    def __init__(self):

        self.feat_dim = 32
        self.out_channels = 8
       

        # GRU disc Configs
        self.disc_hid_dim = 128
        self.disc_out_dim = 1
        self.disc_n_layers = 1
        self.disc_AR_hid = 512
        self.disc_AR_bid = False

        # Transformer Disc Parameters
        self.att_hid_dim = 128
        self.patch_size = self.out_channels
        self.depth =4
        self.heads = 4
        self.mlp_dim = 64
        
        # Cross-domain Training
        self.num_epoch = 15
        self.batch_size = 512 # best is 512



        # data parameters
        self.shuffle = True
        self.drop_last = True
        self.apply_transform = True
        self.num_classes = 3
        self.class_names = ['Healthy', 'D1', 'D2']


        self.base_model = base_model_configs()

        self.SLARDA= SLARD_Configs()

class Supervised(object):
    def __init__(self):

        # training configs
        self.num_epoch = 20
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
        self.out_channels = 8
        self.num_classes = 3
        self.dropout = 0
        self.cls_hidden_dim = 32
        self.feat_dim = 32
        self.kernel_size = 32
        self.cls_hidden_dim = 32
        self.AR = True

# Cross-domain Configs
class SLARD_Configs(object):
    def __init__(self):
        self.optimizer = 'adam'
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.lr = 1e-4
        self.lr_disc = 1e-4
        self.gamma = 0.1
        self.step_size = 50
        self.AR= 'GRU'

        self.teacher_wt  = 0.5 # Teacher only#,  0.005
        self.confidence_level = 0.9
        self.momentum_wt =0.99
