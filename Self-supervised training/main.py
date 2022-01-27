import torch

import os
import numpy as np
from datetime import datetime

from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from args import args


# Args selections
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = args.selected_method
training_mode = args.training_mode
logs_save_dir = args.logs_save_dir
if not os.path.exists(logs_save_dir):
    os.mkdir(logs_save_dir)

exec(f'from models.{data_type}_models import CNN_AR as base_model')

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################




### Pretraining step:
def pretrain_model(src_id):
    # Load Model
    print("====Start Pretraining...")
    configs.training_mode = 'self_supervised'
    model = base_model(configs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2))
    # Trainer
    Trainer(model, optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, src_id)

    # # Testing
    # outs = model_evaluate(model, test_dl, configs, device)

def fine_tune_model(src_id):
    # Load Model
    print("====Start Finetunning...")
    configs.training_mode = 'fine_tune'
    model = base_model(configs).to(device)
    # load saved model of this experiment
    checkpoint_path = os.path.join(os.path.join( "saved_models",  f'last_{args.selected_dataset}_CNN_AR_src_{src_id}_.pt'))
    # load_from = logs_save_dir
    chkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2))

    # Trainer
    Trainer(model, optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, src_id)

    # Testing
    outs = model_evaluate(model, test_dl, configs, device)


if __name__ == "__main__":

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, training_mode)
    os.makedirs(experiment_log_dir, exist_ok=True)

    # loop through domains
    counter = 0
    src_counter = 0

    # Load datasets


    data_path = f"../Autorgressive_Adaptation/data/{data_type}/"
    # find out the domains IDs
    # data_files = os.listdir(data_path)
    # data_files = [i for i in data_files if "train" in i]
    # sources = [i[6] for i in data_files]
    # src_tgt_product = [sources, sources]

    for src_id in ['a', 'b', 'c', 'd']:
        train_dl, valid_dl, test_dl = data_generator(data_path, src_id, configs)
        # Logging
        log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
        logger = _logger(log_file_name)
        logger.debug("=" * 45)
        logger.debug(f'Dataset: {data_type}')
        logger.debug(f'Method:  {method}')
        logger.debug("=" * 45)

        ## Model pretraining
        pretrain_model(src_id)

        # Model finetunning
        fine_tune_model(src_id)





