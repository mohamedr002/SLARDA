import torch
import torch.nn as nn
import wandb
import os
from args import args
from torch.utils.tensorboard import SummaryWriter
from trainer.training_evaluation import model_train, model_evaluate, train_source_only_mixup, model_evaluate_mixup, so_model_evaluate_mixup
from datetime import datetime
import sys
sys.path.append("..")
from utils import weights_init

exec(f'from config_files.{args.selected_dataset}_Configs import Config as Configs')
config = Configs()


# from config_file import domain_adaptation_configs as da_configs


def Trainer(model, train_dl, valid_dl, test_dl, src_id, device, logger):
    ## Start training
    logger.debug("Pretraining_step....")
    best_acc = 0
    best_epoch = -1
    chk_path = f'./src_only_saved_models/{args.selected_dataset}/'
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Supervised.lr, betas=(config.Supervised.beta1, config.Supervised.beta2))
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True,patience=config.plat_patience)

    criterion = nn.CrossEntropyLoss()
    if args.tensorboard:
        comment = (f'./visualize/{src_id}')
        tb = SummaryWriter(comment)
    for epoch in range(1, config.Supervised.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, optimizer, criterion, train_dl, device)
        valid_loss, valid_acc, _, _ = model_evaluate(model, valid_dl, device)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.2f}\t | \tTrain Accuracy     : {train_acc:2.2f}\n'
                     f'Valid Loss     : {valid_loss:.2f}\t | \tValid Accuracy     : {valid_acc:2.2f}')
        if args.tensorboard:
            tb.add_scalar('Train_loss', train_loss, epoch)
            tb.add_scalar(f'Train_accuracy', train_acc, epoch)
            tb.add_scalar(f'Val_loss', valid_loss, epoch)
            tb.add_scalar(f'Val_accuracy', valid_acc, epoch)

            if valid_acc > best_acc:
                best_acc = valid_acc
                chkpoint = {
                    'epoch': epoch + 1,
                    'validation_acc': valid_loss,
                    'model_state_dict': model.state_dict(),
                    'validation_loss': valid_loss}
                if config.Supervised.save_ckp:
                    if not os.path.exists(chk_path):
                        os.mkdir(chk_path)
                    torch.save(chkpoint,
                               f'./src_only_saved_models/{args.selected_dataset}/best_{args.selected_dataset}_{args.base_model}_src_{src_id}.pt')
                best_epoch = epoch + 1
            elif epoch - best_epoch > 2:
                best_epoch = epoch + 1
    if config.Supervised.save_ckp:
        if not os.path.exists(chk_path):
            os.mkdir(chk_path)
        chkpoint = {'model_state_dict': model.state_dict()}
        torch.save(chkpoint,
                   f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_{args.base_model}_src_{src_id}.pt')
    # evaluate on the test set
    logger.debug('\nEvaluate on the Test set:')
    test_loss, test_acc, _, _ = model_evaluate(model, test_dl, device)
    logger.debug(f'Test_loss: {test_loss:0.2f} || Test_acc: {test_acc:0.2f}')

    logger.debug("\n################## Finished Pretraining #########################")
    return model



def Mixup_Trainer(netF, netC, train_dl, valid_dl, test_dl, src_id, device, logger):
    ## Start training
    logger.debug("Pretraining_step....")
    best_acc = 0
    best_epoch = -1
    criterion = nn.CrossEntropyLoss()
    optimizerF = torch.optim.Adam(netF.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizerC = torch.optim.Adam(netC.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    if args.tensorboard:
        comment = (f'./visualize/{src_id}')
        tb = SummaryWriter(comment)
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = train_source_only_mixup(netF, netC, optimizerF, optimizerC, criterion, train_dl, device)
        valid_loss, valid_acc = so_model_evaluate_mixup(netF, netC, valid_dl, device)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.2f}\t | \tTrain Accuracy     : {train_acc:2.2f}\n'
                     f'Valid Loss     : {valid_loss:.2f}\t | \tValid Accuracy     : {valid_acc:2.2f}')
        if args.tensorboard:
            tb.add_scalar('Train_loss', train_loss, epoch)
            tb.add_scalar(f'Train_accuracy', train_acc, epoch)
            tb.add_scalar(f'Val_loss', valid_loss, epoch)
            tb.add_scalar(f'Val_accuracy', valid_acc, epoch)

            if valid_acc > best_acc:
                best_acc = valid_acc
                chkpoint = {
                    'epoch': epoch + 1,
                    'validation_acc': valid_loss,
                    'model_F_state_dict1': netF.state_dict(),
                    'model_C_state_dict1': netC.state_dict(),
                    'validation_loss': valid_loss}
                if config.save_ckp:
                    if not os.path.exists(f'./src_only_saved_models/{args.selected_dataset}/'):
                        os.mkdir(f'./src_only_saved_models/{args.selected_dataset}/')
                    torch.save(chkpoint,
                               f'./src_only_saved_models/{args.selected_dataset}/best_{args.selected_dataset}_src_{src_id}.pt')
                best_epoch = epoch + 1
            elif epoch - best_epoch > 2:
                best_epoch = epoch + 1
    if config.save_ckp:
        if not os.path.exists(f'./src_only_saved_models/{args.selected_dataset}'):
            os.mkdir(f'./src_only_saved_models/{args.selected_dataset}')
        torch.save(netF.state_dict(),
                   f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_netF_src_{src_id}.pt')
        torch.save(netC.state_dict(),
                   f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_netC_src_{src_id}.pt')
    # evaluate on the test set
    logger.debug('\nEvaluate on the Test set:')
    test_loss, test_acc = so_model_evaluate_mixup(netF, netC, test_dl, device)
    logger.debug(f'Test_loss: {test_loss:0.2f} || Test_acc: {test_acc:0.2f}')

    logger.debug("\n################## Finished Pretraining #########################")
    return netF, netC



