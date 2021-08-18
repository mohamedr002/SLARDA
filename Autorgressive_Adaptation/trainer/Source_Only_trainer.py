import torch
import torch.nn as nn
import sys
import os
import wandb

sys.path.append("..")
from trainer.source_only import Trainer
from trainer.training_evaluation import model_evaluate
from args import args
from utils import set_requires_grad, count_parameters, AverageMeter
from torch.utils.tensorboard import SummaryWriter
from models.models import Discriminator_ATT, Discriminator_AR


def cross_domain_train(src_train_dl, src_valid_dl, src_test_dl,
                       tgt_train_dl, tgt_valid_dl, base_model,
                       src_id, tgt_id,
                       device, logger, configs):

    model_configs = configs.base_model

    # source model network.
    source_model = base_model(model_configs).float().to(device)

    # Logging
    logger.debug(f'The model has {count_parameters(source_model):,} trainable parameters')
    logger.debug('=' * 45)
    if args.tensorboard:
        comment = (f'../visualize/{src_id} to {tgt_id}_{args.selected_dataset}')
        tb = SummaryWriter(comment)

    # check if source only model exists, else train it ...
    # ckp_path = f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_{args.base_model}_src_{src_id}.pt'
    # if os.path.exists(ckp_path):
    #     src_chkpoint = torch.load(ckp_path)['model_state_dict']
    # else:
    # Trainer(source_model, src_train_dl, src_valid_dl, src_test_dl, src_id, device, logger)
    ckp_path = f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_{args.base_model}_src_{src_id}.pt'
    src_chkpoint = torch.load(ckp_path)['model_state_dict']

    # Load trained mode;
    source_model.load_state_dict(src_chkpoint)

    source_loss, source_score, _, _ = model_evaluate(source_model, tgt_valid_dl, device)
    logger.debug(f'Src_only Loss : {source_loss:.4f}\t | \tSrc_only Accuracy : {source_score:2.4f}\n')

    return source_model, source_model
