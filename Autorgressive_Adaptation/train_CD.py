import torch
import pandas as pd
import os
from shutil import copy
from utils import fix_randomness, save_to_df, _logger, report_results, get_nonexistant_path,  copy_Files
from dataloader.dataloader import data_generator
from trainer.training_evaluation import cross_domain_test
from datetime import datetime
from itertools import product
from args import args
import wandb
start_time = datetime.now()
device = torch.device(args.device)
da_method = args.da_method
save_dir = args.save_dir
data_type = args.selected_dataset
data_path = f"./data/{data_type}"
base_model_type = args.base_model
experiment_description = args.experiment_description
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

exec(f'from trainer.{da_method} import cross_domain_train')
exec(f'from config_files.{data_type}_Configs import Config as Configs')
exec(f'from models.models import {base_model_type} as base_model')
configs = Configs()

# os.environ["WANDB_MODE"] = "dryrun"
os.environ["WANDB_SILENT"] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# torch.backends.cudnn.enabled = False # another solution for lstm lunch faiulure issue


def main_train_cd():

    # find out the domains IDs
    data_files = os.listdir(data_path)
    data_files = [i for i in data_files if "train" in i]
    sources = [i[6] for i in data_files]
    src_tgt_product = [sources, sources]


    simple_column_names = ['Run ID',
                           'source_loss', 'source_acc',
                            'target_loss', 'target_acc',]
    column_names_mean = ['Scenario',
                         'Source_only_loss_mean', 'Source_only_acc_mean',
                         f'{da_method}_loss_mean', f'{da_method}_acc_mean',
                         f'Source_only_loss_std', 'Source_only_acc_std',
                         f'{da_method}_loss_std', f'{da_method}_acc_std']

    simple_df= pd.DataFrame(columns=simple_column_names)
    mean_df = pd.DataFrame(columns=column_names_mean)
    # Logging
    # cwd = os.getcwd()
    # exp_log_dir = os.path.join(r"D:\Autoregressive Domain Adaptation for Time series data\Last",save_dir, experiment_description, f"{da_method}_{data_type}_{args.run_description}")
    exp_log_dir = os.path.join(os.getcwd(),save_dir, experiment_description, f"{da_method}_{data_type}_{args.run_description}")

    exp_log_dir = get_nonexistant_path(exp_log_dir)
    # os.makedirs(exp_log_dir, exist_ok=True)
    # copy(f"/home/mohamed/SLARADA/config_files/{data_type}_configs.py", f"{exp_log_dir}/{data_type}_configs.py")
    # copy(f"/home/mohamed/SLARADA/trainer/{da_method}.py", f"{exp_log_dir}/{da_method}_script.py")
    # copy("/home/mohamed/SLARADA/args.py",  f"{exp_log_dir}/args.py")
    copy_Files(exp_log_dir, data_type, da_method)
    # loop through domains
    # loop through domains
    counter = 0
    src_counter = 0
    for src_id, tgt_id in product(*src_tgt_product):
    # for src_id in ['a', 'b', 'c']:
    #     for tgt_id in ['a', 'b','c']:
            if src_id != tgt_id:
                # prepare save directory
                # specify number of consecutive runs
                for run_id in range(args.num_runs):
                    fix_randomness(run_id)

                    # Logging
                    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_"+ str(run_id))
                    os.makedirs(log_dir, exist_ok=True)
                    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
                    logger = _logger(log_file_name)
                    logger.debug("=" * 45)
                    logger.debug(f'Dataset: {data_type}')
                    logger.debug(f'Method:  {da_method}')
                    logger.debug("=" * 45)
                    logger.debug(f'Source: {src_id} ---> Target: {tgt_id}')
                    logger.debug(f'Run ID: {run_id}')
                    logger.debug("=" * 45)


                    # Load datasets
                    src_train_dl, src_valid_dl, src_test_dl = data_generator(data_path, src_id, configs)
                    tgt_train_dl, tgt_valid_dl, tgt_test_dl = data_generator(data_path, tgt_id, configs)



                    if args.tensorboard:
                        wandb.init(project="SLARDA",   group = f'{da_method}_{data_type}', name=f'{src_id}_to_{tgt_id}_run_{run_id}', config=configs,
                                   sync_tensorboard=False, reinit=True, dir=r"./visualize/", )

                    source_model, target_model = cross_domain_train(src_train_dl, src_valid_dl, src_test_dl,
                                                                    tgt_train_dl, tgt_valid_dl, base_model,
                                                                    src_id, tgt_id,
                                                                    device, logger, configs)
                    scores = cross_domain_test(source_model, target_model, src_id, tgt_id,
                                               src_train_dl, tgt_train_dl, src_test_dl, tgt_test_dl,
                                               device, log_dir, logger)

                    run_name = f"domain_{src_id}_run_{run_id}"
                    outs = (run_name,) + scores
                    simple_df.loc[counter] = outs
                    counter += 1


                input_data = [f"{src_id}-->{tgt_id}"]
                input_data.extend(simple_df.iloc[-args.num_runs:, 1:].mean().array)
                input_data.extend(simple_df.iloc[-args.num_runs:, 1:].std().array)
                mean_df.loc[src_counter] = input_data
                src_counter += 1


    # Printing and saving final results
    print(simple_df.to_string())
    print(mean_df.to_string())
    printed_results = mean_df[['Scenario', 'Source_only_acc_mean', 'Source_only_acc_std', f'{da_method}_acc_mean', f'{da_method}_acc_std']]
    mean = mean_df[['Source_only_acc_mean', 'Source_only_acc_std', f'{da_method}_acc_mean', f'{da_method}_acc_std']].mean()
    printed_results.loc[len(printed_results)] = mean
    printed_results.at[len(printed_results)-1, 'Scenario'] = 'Average'

    logger.debug(f"Total training time is {datetime.now() - start_time}")

    logger.debug('=' * 45)
    logger.debug(f'Results using: {da_method}')
    logger.debug('=' * 45)
    logger.debug(mean_df.to_string())
    logger.debug(printed_results.to_string())
    print_res_name = os.path.basename(exp_log_dir)
    simple_df.to_excel(f'{exp_log_dir}/full_res_results_{print_res_name}.xlsx')
    printed_results.to_excel(f'{exp_log_dir}/printed_results_{print_res_name}.xlsx')

    if args.tensorboard:
        wandb.log({"Full_results": wandb.Table(dataframe=simple_df)})
        wandb.log({"Printed_results": wandb.Table(dataframe=printed_results)})



if __name__ == "__main__":
    wandb.config = configs
    main_train_cd()

