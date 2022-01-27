import os
import sys

sys.path.append("..")
import numpy as np

from models.loss import NTXentLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args

exec(f'from config_files.{args.selected_dataset}_Configs import Config as Configs')
configs = Configs()

use_SimCLR = args.use_SimCLR


def  Trainer(model, optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, src_id):
    # Start training
    logger.debug("Training started ....")
    best_acc = 0

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, optimizer, criterion, train_dl, config, device)
        valid_loss, valid_acc, _, _ = model_evaluate(model, valid_dl, config, device)
        if config.training_mode=='fine_tune':
            lr_scheduler.step()
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        # if valid_acc > best_acc:
        #     best_acc = valid_acc
        #     chkpoint = {
        #         'epoch': epoch,
        #         'validation_acc': valid_loss,
        #         'model_state_dict': model.state_dict(),
        #         'validation_loss': valid_loss}
        #
        #     os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        #     torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_best.pt'))


    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join( "saved_models",  f'last_{args.selected_dataset}_CNN_AR_src_{src_id}_.pt'))
    if config.training_mode != "self_supervised":
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, test_dl, config, device)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, optimizer, criterion, train_loader, config, device):
    total_loss = []
    total_acc = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # optimizer
        optimizer.zero_grad()


        data, target = data.float().to(device), target.long().to(device)
        output = model(data)

        # compute loss
        if config.training_mode == "self_supervised":
            loss = output
        else:
            predictions, features = output
            loss = criterion(predictions, target)

            total_acc.append(target.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if config.training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, test_dl, config, device):
    model.eval()
    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, target in test_dl:

            data, target = data.float().to(device), target.long().to(device)
            output = model(data)

            # compute loss
            if config.training_mode == "self_supervised":
                loss = output

            else:
                predictions, features = output
                loss = criterion(predictions, target)
                total_acc.append(target.eq(predictions.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())

            if config.training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    if config.training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs
