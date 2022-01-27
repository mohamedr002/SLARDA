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

    # Average meters
    discriminator_accuracies = AverageMeter()
    discriminator_losses = AverageMeter()
    target_model_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    tgt_cls_losses= AverageMeter()
    # source model network.
    source_model = base_model(model_configs).float().to(device)
    target_model = base_model(model_configs).float().to(device)
    teacher_model = base_model(model_configs).float().to(device)

    if configs.SLARDA.AR== 'ATT':
        feature_discriminator = Discriminator_ATT(configs).float().to(device)
    else:
        feature_discriminator = Discriminator_AR(configs).float().to(device)

    # Logging
    logger.debug(f'The model has {count_parameters(source_model):,} trainable parameters')
    logger.debug('=' * 45)
    if args.tensorboard:
        comment = (f'../visualize/{src_id} to {tgt_id}_{args.selected_dataset}')
        tb = SummaryWriter(comment)

    # check if source only model exists, else train it ...
    ckp_path = f'./src_only_saved_models/{args.selected_dataset}/last_{args.selected_dataset}_{args.base_model}_src_{src_id}.pt'
    if os.path.exists(ckp_path):
        src_chkpoint = torch.load(ckp_path)['model_state_dict']
    # else:
    #     Trainer(source_model, src_train_dl, src_valid_dl, src_test_dl, src_id, device, logger)
    #     src_chkpoint = torch.load(ckp_path)['model_state_dict']

    # Load trained mode;
    source_model.load_state_dict(src_chkpoint)
    target_model.load_state_dict(src_chkpoint)
    teacher_model.load_state_dict(src_chkpoint)

    # Freeze the source domain model
    set_requires_grad(source_model, requires_grad=False)
    set_requires_grad(teacher_model, requires_grad=False)

    # loss functions
    criterion = nn.CrossEntropyLoss()
    criterion_disc = nn.BCEWithLogitsLoss()
    softmax= nn.Softmax(dim=1)


    # losses wt
    loss_tgt_wt = configs.SLARDA.teacher_wt
    step_size = configs.SLARDA.step_size
    gamma = configs.SLARDA.gamma
    confidence_level = configs.SLARDA.confidence_level
    momentum_update = configs.SLARDA.momentum_wt


    # optimizer.
    optimizer_encoder = torch.optim.Adam(target_model.parameters(), lr=configs.SLARDA.lr,
                                         betas=(configs.SLARDA.beta1, configs.SLARDA.beta2),weight_decay=3e-4)
    optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=configs.SLARDA.lr_disc,
                                      betas=(configs.SLARDA.beta1, configs.SLARDA.beta2), weight_decay=3e-4)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=step_size, gamma=gamma)

    # training..
    for epoch in range(1, configs.num_epoch + 1):
        joint_loaders = enumerate(zip(src_train_dl, tgt_train_dl))
        target_model.train()
        feature_discriminator.train()
        n_correct = 0

        for step, ((source_data, source_labels), (target_data, target_labels)) in joint_loaders:
            source_data, source_labels, target_data, target_labels = source_data.float().to(device), source_labels.to(
                device), target_data.float().to(device), target_labels.to(device)

            ###########################
            # train discriminator #
            ###########################

            # zero gradient for the dicriminator
            optimizer_disc.zero_grad()

            # pass data  through the model network.
            source_pred, (source_latent,source_feat) = source_model(source_data)

            # pass images through the target model network.
            pred_target, (target_latent,target_feat) = target_model(target_data)

            # concatenate source and target features


            feat_concat = torch.cat((source_feat, target_feat), dim=0)

            # predict the domain label by the discirminator network
            pred_concat = feature_discriminator(feat_concat.detach())

            # prepare real labels for the training the discriminator
            label_src = torch.ones(source_feat.size(0)).to(device)
            label_tgt = torch.zeros(target_feat.size(0)).to(device)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # Discriminator Loss
            loss_disc = criterion_disc(pred_concat.squeeze(), label_concat.float())
            loss_disc.backward()

            # Update disciriminator optimizer
            optimizer_disc.step()
            # Discriminator accuracy
            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################
            optimizer_disc.zero_grad()
            optimizer_encoder.zero_grad()

            # # Extract target domain features

            pred = source_pred.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(source_labels.data.view_as(pred)).cpu().sum()


            pred_tgt = feature_discriminator(target_feat)

            # prepare fake labels
            label_tgt = (torch.ones(target_feat.size(0))).to(device)

            # compute loss for target encoder
            loss_tgt = criterion_disc(pred_tgt.squeeze(), label_tgt.float())

            # mean teacher model
            with torch.no_grad():
                mean_t_pred, mean_t_output = teacher_model(target_data)
                normalized_pred = softmax(mean_t_pred)
                pred_prob = normalized_pred.max(1, keepdim=True)[0].squeeze()

            target_pseudo_labels = normalized_pred.max(1, keepdim=True)[1].squeeze()
            confident_feat = target_latent[pred_prob > confidence_level]
            confident_pred= pred_target[pred_prob > confidence_level]
            confident_labels = target_pseudo_labels[pred_prob > confidence_level]

            # target_pseudo_labels = pred_target.max(1, keepdim=True)[1].squeeze()
            feat_concat = torch.cat((source_latent, confident_feat), dim=0)

            # Target Psuedo labeling
            loss_cls_tgt= criterion(confident_pred,confident_labels)


            total_loss = loss_tgt + loss_tgt_wt*loss_cls_tgt

            # Average updates
            discriminator_accuracies.update(acc, feat_concat.size(0))
            discriminator_losses.update(loss_disc, feat_concat.size(0))
            target_model_losses.update(loss_tgt, target_data.size(0))
            tgt_cls_losses.update(loss_cls_tgt,confident_pred.size(0))

            # Backpropagate the loss.
            total_loss.backward()

            # optimize target encoder
            optimizer_encoder.step()
            optimizer_disc.step()


            alpha = momentum_update
            for mean_param, param in zip(teacher_model.parameters(), target_model.parameters()):
                mean_param.data.mul_(alpha).add_(1 - alpha, param.data)


        scheduler_encoder.step()

        # Logging
        logger.debug(f'\nEpoch : {epoch}\n')
        logger.debug(f'\t Discriminator_acc \t| {discriminator_accuracies.avg:.4f}\t     ')
        logger.debug(f'\t Discriminator_loss \t| {discriminator_losses.avg:.4f}\t     ')
        logger.debug(f'\t Target_loss \t      | {target_model_losses.avg:.4f}\t')
        logger.debug(f'\t Target_cls_loss \t      | {tgt_cls_losses.avg:.4f}\t')

        if args.tensorboard:
            tb.add_scalar('Discriminator_loss', discriminator_losses.avg, epoch)
            tb.add_scalar('Feature_extractor_loss', target_model_losses.avg, epoch)
            wandb.log({"Discriminator loss": discriminator_losses.avg, "Feature_extractor loss": target_model_losses.avg}, step=epoch)

        if epoch % 1 == 0:
            source_loss, source_score, _, _ = model_evaluate(source_model, tgt_valid_dl, device)
            target_loss, target_score, _, _ = model_evaluate(target_model, tgt_valid_dl, device)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Src_only Loss : {source_loss:.4f}\t | \tSrc_only Accuracy : {source_score:2.4f}\n'
                         f'{args.da_method} Loss     : {target_loss:.4f}\t | \t{args.da_method} Accuracy     : {target_score:2.4f}')
            if args.tensorboard:
                tb.add_scalar('train_loss/Source_only', source_loss)
                tb.add_scalar(f'train_loss/{args.da_method}', target_loss)
                tb.add_scalar('train_accuracy/Source_only', source_score)
                tb.add_scalar(f'train_accuracy/{args.da_method}', target_score)
                wandb.log({"train/Source_only_loss": source_loss, "train/Source_only_acc": source_score})
                wandb.log({f"train/{args.da_method}_loss": target_loss, f"train/{args.da_method}_acc": target_score})

    return source_model, target_model
