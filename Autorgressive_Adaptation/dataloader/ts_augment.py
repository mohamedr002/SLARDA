import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


def add_noise(signal, noise_amount):
    """
    adding noise
    """
    signal = signal.cpu().numpy()
    noise = np.random.normal(1, noise_amount, np.shape(signal)[0])
    noised_signal = signal + noise
    return torch.from_numpy(noised_signal)


def negate(signal):
    """
    negate the signal
    """

    negated_signal = signal * (-1)
    return negated_signal


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    x = x.squeeze()
    x =  x + torch.normal(mean=2, std = sigma, size=(x.shape[0], x.shape[1]) )
    # https://arxiv.org/pdf/1706.00527.pdf
    return x


def scaling(x, sigma=1.1):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    factor = torch.normal(mean=2, std = sigma, size=(x.shape[0], x.shape[2]) )
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(xi  * factor)
    return torch.cat((ai))

def permutation(x, max_segments=5, seg_mode="random"):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def time_shift(x, shift_ratio=0.2):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    signal_length = x.shape[2]
    shift = int(signal_length * shift_ratio)
    shifted_sample = np.concatenate((x[:, :, signal_length-shift:], x[:, :, :signal_length-shift]), axis=2)
    return torch.from_numpy(shifted_sample)


def apply_transformation(X_train):
    #X_train = X_train.detach().cpu().numpy()
    if not torch.is_tensor(X_train):
        X_train = torch.from_numpy(X_train)
    # if len(X_train.shape) <3:
    #     X_train = torch.unsqueeze(X_train, 0)f
    X_train = X_train.squeeze()
    X0 = vat_noise(X_train).float()


    X1 = negate(X_train).squeeze().float()


    X4 = hor_filp(X_train).squeeze().float()


    X5 = time_shift(X_train).squeeze().float()


    # X_aug = torch.from_numpy(np.concatenate((X0, X1, X2, X3), axis=0)).float()
    # y_aug = torch.from_numpy(np.concatenate((y0, y1, y2, y3))).long()

    # return [X0, X1, X4, X5]
    return [X5, X5, X5, X5]

# def apply_transformation(X_train):
#     #X_train = X_train.detach().cpu().numpy()
#     if not torch.is_tensor(X_train):
#         X_train = torch.from_numpy(X_train)
#     if len(X_train.shape) <3:
#         X_train = torch.unsqueeze(X_train, 0)
#
#     X0 = X_train.squeeze(0).float()
#     y0 = torch.from_numpy(np.array([0] * len(X_train))).long()
#
#     X1 = negate(X_train).squeeze(0).float()
#     y1 = torch.from_numpy(np.array([1] * len(X_train))).long()
#
#     X2 = permutation(X_train, max_segments=5).squeeze(0).float()
#     y2 = torch.from_numpy(np.array([2] * len(X_train))).long()
#
#     X3 = scaling(X_train, 2).squeeze(0).float()
#     y3 = torch.from_numpy(np.array([3] * len(X_train))).long()
#
#     X4 = hor_filp(X_train).squeeze(0).float()
#     y4 = torch.from_numpy(np.array([4] * len(X_train))).long()
#
#     X5 = add_noise(X_train, 2).squeeze(0).float()
#     y5 = torch.from_numpy(np.array([5] * len(X_train))).long()
#
#     # X_aug = torch.from_numpy(np.concatenate((X0, X1, X2, X3), axis=0)).float()
#     # y_aug = torch.from_numpy(np.concatenate((y0, y1, y2, y3))).long()
#
#     return [X0, X1, X2, X3, X4, X5], [y0, y1, y2, y3, y4, y5]

def hor_filp(signal):
    """
    flipped horizontally
    """
    if len(signal.shape) == 2:
        signal = signal.unsqueeze(0)
    hor_flipped = np.flip(signal.cpu().numpy(), axis=2)
    return torch.from_numpy(hor_flipped.copy())

import torch.nn.functional as F
def vat_noise(signal, XI=1e-6):
    d = torch.empty(signal.shape).normal_(mean=signal.mean(),std=signal.std())
    d = F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())
    d = XI * d
    return signal + d