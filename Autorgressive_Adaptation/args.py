import os
import argparse
parser = argparse.ArgumentParser()



home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Proposed', type=str,
                    help='Experiment Description:Proposed_Method, Benchmarking, EEG, Cont_DA_tune,  Benchmarking, Src_guided_trial,  SL_DA_tied')
parser.add_argument('--run_description', default='', type=str,
                    help='run_description : ')
parser.add_argument('--save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--da_method', default='SLARDA', type=str,
                    help='Domain Adaptation method:DAN, SLARDA,SLARDA_inv, Source_Only_trainer')
parser.add_argument('--base_model', default='CNN_Opp_HAR_SL', type=str,
                    help='The Base Feature extractor to be used: CNN_Opp_HAR_SL, CNN_SL_bn, EEG_M_SL')
parser.add_argument('--selected_dataset', default='Opp_HAR', type=str,
                    help='Dataset of choice:  Paderborn_FD, Opp_HAR, EEG')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--num_runs', default=3, type=int,
                    help='Number of consecutive run with different seeds')
parser.add_argument('--tensorboard', default=False, type=bool,
                    help='Tensorboard visualization')
parser.add_argument('--seed', default=0, type=float,
                    help='Tensorboard visualization')
parser.add_argument('--plot_tsne', default=False, type=bool,
                    help='Plot t-sne for training and testing or not?')
args = parser.parse_args()