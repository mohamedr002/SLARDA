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
from dataloader.ts_augment import scaling, jitter, permutation


def cross_domain_train(src_train_dl, src_valid_dl, src_test_dl,
                       tgt_train_dl, tgt_valid_dl, base_model,
                       src_id, tgt_id,
                       device, logger, configs):
    model_configs = configs.base_model

    target_model = base_model(model_configs).float().to(device)
    teacher_model = base_model(model_configs).float().to(device)

    set_requires_grad(teacher_model, requires_grad=False)

    # loss functions
    criterion = nn.CrossEntropyLoss()
    criterion_teacher = nn.BCEWithLogitsLoss()
    softmax = nn.Softmax(dim=1)

    # losses wt
    loss_tgt_wt = configs.Self_ensemble.teacher_wt
    step_size = configs.Self_ensemble.step_size
    gamma = configs.Self_ensemble.gamma
    confidence_level = configs.Self_ensemble.confidence_level
    momentum_update = configs.Self_ensemble.momentum_wt

    scale_ratio = 0.05
    max_seg = 5
    jitter_ratio = 0.1

    def DataTransform(sample, scale_ratio , max_seg, jitter_ratio):
        weak_aug = scaling(sample, scale_ratio)
        strong_aug = jitter(sample, jitter_ratio)
        # strong_aug = jitter(permutation(sample, max_segments=max_seg), jitter_ratio)

        return weak_aug, strong_aug

    # optimizer.
    optimizer_encoder = torch.optim.Adam(target_model.parameters(), lr=configs.Self_ensemble.lr,
                                         betas=(configs.SLARDA.beta1, configs.SLARDA.beta2), weight_decay=3e-4)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=step_size, gamma=gamma)

    # training..
    for epoch in range(1, configs.num_epoch + 1):
        joint_loaders = enumerate(zip(src_train_dl, tgt_train_dl))
        target_model.train()

        for step, ((source_data, source_labels), (target_data, target_labels)) in joint_loaders:

            # pass images through the target model network.
            student_aug, teacher_aug = DataTransform(target_data,  scale_ratio, max_seg, jitter_ratio)
            source_data, source_labels, target_data, target_labels, student_aug, teacher_aug = source_data.float().to(device), source_labels.to(
                device), target_data.float().to(device), target_labels.to(device), student_aug.to(device), teacher_aug.to(device)

            # pass data  through the model network.
            source_pred, (source_feat, _) = target_model(source_data)

            pred_target, (target_feat, _) = target_model(student_aug)

            # Task classification  Loss
            src_cls_loss = criterion(source_pred.squeeze(), source_labels)

            # mean teacher model
            with torch.no_grad():
                mean_t_pred, mean_t_output = teacher_model(teacher_aug)
                teacher_prob = softmax(mean_t_pred)
                top_prob, top_class = teacher_prob.topk(1, dim=1)

            student_prob = softmax(pred_target)
            teacher_predictions = teacher_prob[top_prob.squeeze() > confidence_level]
            student_predictions = student_prob[top_prob.squeeze() > confidence_level]

            loss_cls_tgt = criterion_teacher(student_predictions, teacher_predictions)

            total_loss = src_cls_loss + loss_tgt_wt * loss_cls_tgt

            # Backpropagate the loss.
            total_loss.backward()

            # optimize target encoder
            optimizer_encoder.step()

            alpha = momentum_update
            for mean_param, param in zip(teacher_model.parameters(), target_model.parameters()):
                mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

        scheduler_encoder.step()

        if epoch % 1 == 0:
            target_loss, target_score, _, _ = model_evaluate(target_model, tgt_valid_dl, device)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'{args.da_method} Loss     : {target_loss:.4f}\t | \t{args.da_method} Accuracy     : {target_score:2.4f}')

    return target_model, target_model
