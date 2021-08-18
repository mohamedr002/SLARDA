import torch
from torch import nn

import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

### For tsne
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.autograd import Variable

from args import args


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        # if name=='weight':
        #     nn.init.kaiming_uniform_(param.data)
        # else:
        #     torch.nn.init.zeros_(param.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def mean_std(x):
    mean = np.mean(np.array(x))
    std = np.std(np.array(x))
    return mean, std


def save_to_df_1(run_id, data_id, scores):
    res = []
    for metric in scores:
        mean = np.mean(np.array(metric))
        std = np.std(np.array(metric))
        res.append(f'{mean:2.2f}')
        res.append(f'{std:2.2f}')
    df_out = pd.Series((run_id, data_id, res), index = df.columns)
    # df_out = pd.Series((run_id, data_id, res[0][0],res[0][1],res[1][0],res[1][1],res[2][0],res[2][1],res[2][0],res[2][1],res[3][0],res[3][1]))
    return df_out
def save_to_df(scores):
    res = []
    for metric in scores:
        mean = np.mean(np.array(metric))
        std = np.std(np.array(metric))
        res.append(f'{mean:2.2f}')
        res.append(f'{std:2.2f}')
    # df_out = pd.Series((run_id, data_id, res[0][0],res[0][1],res[1][0],res[1][1],res[2][0],res[2][1],res[2][0],res[2][1],res[3][0],res[3][1]))
    return res


def _calc_metrics(pred_labels, true_labels, log_dir):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.mkdir(labels_save_path)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    file_name = os.path.basename(os.path.normpath(log_dir)) + "_classification_report.xlsx"
    report_Save_path = os.path.join(args.home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = os.path.basename(os.path.normpath(log_dir)) + "_confusion_matrix.torch"
    cm_Save_path = os.path.join(args.home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)



def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def _plot_tsne(model, src_dl, tgt_dl, device, save_dir, model_type, train_mode): #, layer_output_to_plot, y_test, save_dir, type_id):
    print("Plotting TSNE for " + model_type +"...")

    with torch.no_grad():
        src_data = src_dl.dataset.x_data.float().to(device)
        src_labels = src_dl.dataset.y_data.view((-1)).long() #.to(device)
        src_predictions, src_features = model(src_data)

        tgt_data = tgt_dl.dataset.x_data.float().to(device)
        tgt_labels = tgt_dl.dataset.y_data.view((-1)).long() #.to(device)
        tgt_predictions, tgt_features = model(tgt_data)



    perplexity = 50
    src_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform(
        (Variable(src_features).data).detach().cpu().numpy().reshape(len(src_labels), -1).astype(np.float64))

    tgt_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform(
        (Variable(tgt_features).data).detach().cpu().numpy().reshape(len(tgt_labels), -1).astype(np.float64))

    fig, ax = plt.subplots(figsize=(16, 10))
    cmaps = plt.get_cmap('jet')
    src_scatter = ax.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=20, c=src_labels, cmap=cmaps,
                             label="source data")
    tgt_scatter = ax.scatter(tgt_model_tsne[:, 0], tgt_model_tsne[:, 1], s=20, c=tgt_labels, cmap=cmaps,
                             label="target data", marker='^')

    legend1 = ax.legend(*src_scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    ax.legend()



    if not os.path.exists(os.path.join(save_dir, "tsne_plots")):
        os.mkdir(os.path.join(save_dir, "tsne_plots"))

    file_name = "tsne_" + model_type + "_" + train_mode + ".png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    plt.savefig(fig_save_name)
    plt.close()
    # '[0]*len(src_labels), cmap=colors,
    fig, ax = plt.subplots(figsize=(16, 10))
    # colors = ['Greens_r', 'Blues_r']
    ax.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=10, c='red' ,
                             label="source data")
    ax.scatter(tgt_model_tsne[:, 0], tgt_model_tsne[:, 1], s=10, c='blue',
                             label="target data")
    ax.legend()

    file_name = "tsne_" + model_type + "_" + train_mode + "_domain-based.png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    plt.savefig(fig_save_name)
    plt.close()


def plot_tsne_one_domain(model, src_dl, device, save_dir, model_type, train_mode):

    with torch.no_grad():
        src_data = src_dl.dataset.x_data.float().to(device)
        src_labels = src_dl.dataset.y_data.view((-1)).long()  # .to(device)
        src_predictions, src_features = model(src_data)

    perplexity = 50
    src_model_tsne = TSNE(n_components=2, random_state=1, perplexity=perplexity).fit_transform((Variable(src_features).data).detach().cpu().numpy().reshape(len(src_labels), -1).astype(np.float64))


    fig, ax = plt.subplots(figsize=(16, 10))
    cmaps = plt.get_cmap('jet')
    src_scatter = ax.scatter(src_model_tsne[:, 0], src_model_tsne[:, 1], s=20, c=src_labels, cmap=cmaps,
                             label="source data")

    legend1 = ax.legend(*src_scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax.add_artist(legend1)
    ax.legend()
    if not os.path.exists(os.path.join(save_dir, "tsne_plots")):
        os.mkdir(os.path.join(save_dir, "tsne_plots"))

    file_name = "tsne_" + model_type + "_" + train_mode + ".png"
    fig_save_name = os.path.join(save_dir, "tsne_plots", file_name)
    plt.savefig(fig_save_name)
    plt.close()