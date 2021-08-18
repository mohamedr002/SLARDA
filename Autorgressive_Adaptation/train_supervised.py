#
import torch
import pandas as pd
import os
# import wandb
from utils import fix_randomness, save_to_df, _logger
from dataloader.dataloader import data_generator
from trainer.training_evaluation import same_domain_test
from datetime import datetime
from args import args
from trainer.source_only import Trainer
from models.models import EEG_M as base_Model

data_type = args.selected_dataset

device = torch.device(args.device)
exec(f'from config_files.{data_type}_Configs import Config as Configs')

configs = Configs()
model_configs = configs.base_model

save_dir = args.save_dir
data_type = args.selected_dataset
data_path = f"./data/{data_type}"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# find out the domains IDs
data_files = os.listdir(data_path)
data_files = [i for i in data_files if "train" in i]
sources = [i[6] for i in data_files]

# define data frame
# logging

# logging
column_names = ['Run ID',
                'train_loss', 'train_acc',
                'val_loss', 'val_acc',
                'test_loss', 'test_acc']
column_names_mean = ['Domain ID',
                     'train_loss_mean','train_acc_mean',
                     'val_loss_mean','val_acc_mean',
                     'test_loss_mean','test_acc_mean',
                     'train_loss_std','train_acc_std',
                     'val_loss_std', 'val_acc_std',
                     'test_loss_std', 'test_acc_std']

full_res_df = pd.DataFrame(columns=column_names)
mean_df = pd.DataFrame(columns=column_names_mean)

exp_log_dir = datetime.now().strftime('%d_%m_%Y_%H_%M_%S_') + 'supervised' + "_" + data_type
os.mkdir(os.path.join(save_dir, exp_log_dir))

# loop through domains
counter = 0
src_counter = 0
for src_id in sources:

    # Load datasets
    src_train_dl, src_valid_dl, src_test_dl = data_generator(data_path, src_id, configs)

    full_train_score, full_test_score = [], []

    # load model

    # specify number of consecutive runs
    for run_id in range(args.num_runs):
        # Logging
        log_dir = os.path.join(save_dir, exp_log_dir, src_id + str(run_id))
        os.mkdir(log_dir)
        log_file_name = os.path.join(log_dir, "info.log")
        logger = _logger(log_file_name)
        logger.debug("=" * 45)
        logger.debug(f'Dataset: {data_type}')
        logger.debug(f'Method:  Supervised')
        logger.debug("=" * 45)
        logger.debug(f'Domain_id: {src_id}')
        logger.debug("=" * 45)

        # Load Model
        model = base_Model(model_configs).float().to(device)

        # Trainer
        model = Trainer(model, src_train_dl, src_valid_dl, src_test_dl, src_id, device, logger)

        # Eval and Testing
        outs = same_domain_test(model, src_id,
                                src_train_dl, src_valid_dl, src_test_dl,
                                device, log_dir, logger
                                )
        # Logging to dataframe
        run_name = f"domain_{src_id}_run_{run_id}"
        outs = (run_name,) + outs
        full_res_df.loc[counter] = outs
        counter += 1
    input_data = [f"Domain_{src_id}"]
    input_data.extend(full_res_df.iloc[-args.num_runs:, 1:].mean().array)
    input_data.extend(full_res_df.iloc[-args.num_runs:, 1:].std().array)
    mean_df.loc[src_counter] = input_data
    src_counter += 1

# Printing and saving final results
print(full_res_df.to_string())
print(mean_df.to_string())
mean_df.to_csv(f'{os.path.join(save_dir, exp_log_dir)}/mean_results.csv')
full_res_df.to_csv(f'{os.path.join(save_dir, exp_log_dir)}/full_res_results.csv')


