import torch
import torch.nn.functional as F
import numpy as np
from args import args
from utils import _calc_metrics,plot_tsne_one_domain, _plot_tsne
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
data_type = args.selected_dataset
def model_train(model, optimizer, criterion,  train_loader, device):
    total_loss = []
    total_acc = []
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        # send to device
        data = data.float().to(device)
        labels = labels.view((-1)).long().to(device)

        # optimizer
        optimizer.zero_grad()

        # forward pass
        predictions, features = model(data)

        # compute loss
        loss = criterion(predictions, labels)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        # pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
        total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())


    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc

def train_source_only_mixup(netF, netC, optimizerF, optimizerC, criterion, data_loader, device):
    total_loss = []
    total_acc = []

    netF.train()
    netC.train()

    for inputs, labels in data_loader:
        inputs, labels = inputs.float().to(device), labels.long().to(device)


        netC.zero_grad()
        netF.zero_grad()
        outC = netC(netF(inputs))
        loss = criterion(outC, labels)
        total_loss.append(loss.item())
        loss.backward()
        optimizerC.step()
        optimizerF.step()
        total_acc.append(labels.eq(outC.detach().argmax(dim=1)).float().mean())

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean() * 100
    return total_loss, total_acc


def model_evaluate(model, valid_dl, device):
    model.eval()
    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels in valid_dl:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            # compute loss
            loss = criterion(predictions, labels)
            total_loss.append(loss.item())
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # total_acc += pred.eq(target.view_as(pred)).sum().item()
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

    # total_loss /= len(valid_dl.dataset)  # average loss
    # total_acc /= 1. * len(valid_dl.dataset)  # average acc

    total_loss = torch.tensor(total_loss).mean() # average loss
    total_acc = torch.tensor(total_acc).mean()   #average acc
    return total_loss, total_acc, outs, trgs


####### START: FOR Domain_mixup ############################
def so_model_evaluate_mixup(netF, netC, valid_dl, device):
    netF.eval()
    netC.eval()
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    # Testing the model
    with torch.no_grad():
        for data, labels in valid_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            outC = netC(netF(data))
            loss = criterion(outC, labels)
            _, predicted = torch.max(outC.data, 1)
            total += labels.size(0)
            total_loss.append(loss)
            correct += ((predicted == labels.to(device)).sum())

        total_loss = torch.tensor(total_loss).mean()
        val_acc = 100 * float(correct) / total
    return total_loss, val_acc

def model_evaluate_mixup(netF, netC, valid_dl, device):
    netF.eval()
    netC.eval()
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    # Testing the model
    with torch.no_grad():
        for data, labels in valid_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            embedding, mean, std = netF(data)
            mean_std = torch.cat((mean, std), 1)
            outC_logit, _ = netC(mean_std)
            loss = criterion(outC_logit, labels)
            total_loss.append(loss.item())

            _, predicted = torch.max(outC_logit.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels.to(device)).sum())

        total_loss = torch.tensor(total_loss).mean()
        val_acc = 100 * float(correct) / total
    return total_loss, val_acc
####### END: FOR Domain_mixup ############################



def da_validate(model, classifier, valid_dl, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data, target in valid_dl:
            data = data.float().to(device)
            target = target.view((-1,)).long().to(device)
            hidden = model.init_hidden(len(data))
            output, hidden = model.predict(data, hidden)
            output = classifier(output[:, -1, :])
            total_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(valid_dl.dataset)  # average loss
    total_acc /= 1. * len(valid_dl.dataset)  # average acc
    print('Validation loss: {:.4f}\t Validation Accuracy: {:.4f}\n'.format(total_loss, total_acc))
    return total_loss, total_acc

def cross_domain_test(source_model, target_model,src_id, tgt_id,
                      src_train_dl, tgt_train_dl,src_test_dl, tgt_test_dl,
                      device, log_dir,logger):
    if args.tensorboard:
        comment = (f'./visualize/{src_id} to {tgt_id}_{data_type}')
        tb = SummaryWriter(comment)
    if args.plot_tsne:
        _plot_tsne(source_model, src_train_dl, tgt_train_dl, device, log_dir, 'src_only', 'train')
        _plot_tsne(target_model, src_train_dl, tgt_train_dl, device, log_dir, f'{args.da_method}', 'train')

        _plot_tsne(source_model, src_test_dl, tgt_test_dl, device, log_dir, 'src_only', 'test')
        _plot_tsne(target_model, src_test_dl, tgt_test_dl, device, log_dir, f'{args.da_method}', 'test')

    # finish Training evaluate on test sets
    logger.debug('==== Domain Adaptation completed =====')
    logger.debug('\n==== Evaluate on test sets ===========')
    source_loss, source_score, _, _ = model_evaluate(source_model, tgt_test_dl, device)
    target_loss, target_score, pred_labels, true_labels = model_evaluate(target_model, tgt_test_dl, device)
    _calc_metrics(pred_labels, true_labels, log_dir)

    logger.debug(f'\t Src_only Loss : {source_loss:.4f}\t | \tSrc_only Accuracy : {source_score:2.4f}')
    logger.debug(f'\t {args.da_method} Loss     : {target_loss:.4f}\t | \t{args.da_method} Accuracy     : {target_score:2.4f}')
    # wandb.sklearn.plot_confusion_matrix(true_labels, pred_labels, configs['class_names'])

    if args.tensorboard:
        tb.add_scalar('test/Source_only_loss', source_loss)
        tb.add_scalar(f'test/{args.da_method}_loss', target_loss)
        tb.add_scalar('test/Source_only_accuracy', source_score)
        tb.add_scalar(f'test/{args.da_method}_accuracy/', target_score)
        wandb.log({"test/Source_only_loss": source_loss, "test/Source_only_acc": source_score})
        wandb.log({f"test/{args.da_method}_loss": target_loss, f"test/{args.da_method}_acc": target_score})

    return source_loss.item(), source_score.item()*100, target_loss.item(), target_score.item()*100


def same_domain_test(source_model,src_id,src_train_dl,src_valid_dl, src_test_dl,device, log_dir,logger):
    if args.tensorboard:
        comment = (f'./visualize/domain_{src_id}_{data_type}')
        tb = SummaryWriter(comment)
    if args.plot_tsne:
        plot_tsne_one_domain(source_model, src_train_dl, device, log_dir, 'Supervised', 'train')
        plot_tsne_one_domain(source_model, src_test_dl, device, log_dir, 'Supervised', 'test')


    # finish Training evaluate on test sets
    logger.debug('==== Supervised Performance =====')
    logger.debug('==== Evaluate on test sets ===========')
    train_loss, train_score,  _, _ = model_evaluate(source_model, src_train_dl, device)
    val_loss, val_score,  _, _ = model_evaluate(source_model, src_valid_dl, device)
    test_loss, test_score,  pred_labels, true_labels = model_evaluate(source_model, src_test_dl, device)
    _calc_metrics(pred_labels, true_labels, log_dir)
    logger.debug(f'\t Train Loss : {train_loss:.2f}\t | \tTrain Accuracy : {train_score:2.4f}')
    logger.debug(f'\t Val Loss : {val_loss:.2f}\t | \tVal Accuracy : {val_score:2.4f}')
    logger.debug(f'\t Test Loss : {test_loss:.2f}\t | \tTest Accuracy : {test_score:2.4f}')

    # wandb.sklearn.plot_confusion_matrix(true_labels, pred_labels, configs['class_names'])

    if args.tensorboard:
        tb.add_scalar('same_domain/test_loss', test_loss)
        tb.add_scalar(f'same_domain/test_accuracy', test_score)
        wandb.log({"same_domain/test_loss ": test_loss, "same_domain/test_accuracy": test_score})
    return  train_loss.item(), train_score.item(), val_loss.item(), val_score.item(), test_loss.item(), test_score.item()