import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib


# n_power= 1
# radius = 3.5


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


def confidence_thresholding(logits):
    a, b = torch.topk(logits, 2)
    d = torch.where(a[:, 0] - a[:, 1] >= 0.2)
    return d


class NCE_model(nn.Module):
    def __init__(self, device, delta):
        super(NCE_model, self).__init__()
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.delta = delta
        self.device = device

    def forward(self, src_feas, src_labels, tgt_feas, tgt_logits):
        nce = 0
        batch_size = src_feas.size(0)
        src_index, tgt_index = self.extract_positive_pairs(src_feas, src_labels, tgt_feas, tgt_logits, self.delta)
        # src_feas--> output features of source feature extractor #dim (batch_size, feature_dim)
        # tgt_feas--> output features of target feature extractor #dim (batch_size, feature_dim)
        z_i = src_feas[src_index]
        z_j = tgt_feas[tgt_index].permute(1, 0)

        total = torch.mm(z_i, z_j)  # e.g. size 8*8
        nce = torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch_size * len(src_index)
        return nce

    def extract_positive_pairs(self, src_feas, src_labels, tgt_feas, tgt_logits, delta):
        # delta--> confidence thresholding parameter 
        # Extract confident logits 
        v, i = torch.topk(tgt_logits, 2)
        # condition for thresholding confident classes
        condition = (v[:, 0] - v[:, 1] >= delta)
        # indices for conifdent classes
        confident_indices = torch.where(condition)
        # tuple to tensor
        confident_indices = confident_indices[0]
        # get the confident classes it self 
        confident_class = tgt_logits[confident_indices].argmax(1)
        # Combining different pairs 
        pos_pairs_indices = {}
        pos_pairs = {}
        full_src_pairs, full_tgt_pairs, full_src_idx, full_tgt_idx = [], [], [], []
        # for each class find the source and target pairs 
        for i, v in enumerate(confident_class):
            # find indices where source labels equal to conifdent classes of target
            src_index = (torch.where(src_labels == v))
            if len(src_index) == 0:
                continue
            src_index = src_index[0]
            # indexing condint classes position of target 
            tgt_index = confident_indices[i].repeat(len(src_index))
            # positive pairs between source and target indices 
            pos_pairs_indices[v] = src_index, tgt_index
            # extract the corresponding samples for the source 
            src_pair = src_feas[src_index]
            # extract the corresponding samples for the target 
            tgt_pair = tgt_feas[tgt_index].reshape(src_pair.size())
            if src_pair.size(0) != 0:
                pos_pairs[v] = src_pair, tgt_pair
                # construct the positive pairs
                full_src_pairs.append(src_pair)
                full_tgt_pairs.append(tgt_pair)
                full_src_idx.append(src_index)
                full_tgt_idx.append(tgt_index)
        # # Convert list of tensors to tensors 
        # src_pair_tensor = torch.cat((full_src_pairs), dim=0,) 
        # tgt_pair_tensor = torch.cat((full_tgt_pairs), dim=0,)
        src_pair_idx = torch.cat((full_src_idx), dim=0, )
        tgt_pair_idx = torch.cat((full_tgt_idx), dim=0, )

        return src_pair_idx, tgt_pair_idx

class my_cntrst_loss(nn.Module):
    def __init__(self, device, delta):
        super(my_cntrst_loss, self).__init__()
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.delta = delta
        self.device = device

    def forward(self, src_feas, src_labels, tgt_feas, tgt_logits):
        nce = 0
        batch_size = src_feas.size(0)
        src_index, tgt_index = self.extract_positive_pairs(src_feas, src_labels, tgt_feas, tgt_logits, self.delta)
        # src_feas--> output features of source feature extractor #dim (batch_size, feature_dim)
        # tgt_feas--> output features of target feature extractor #dim (batch_size, feature_dim)
        z_i = src_feas[src_index]
        z_j = tgt_feas[tgt_index].permute(1, 0)
        pos_size = z_i.size(0)
        # [2*B, D]
        total_pos = torch.cat([z_i, z_j], dim=0)

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(total_pos, total_pos.t().contiguous()))
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1))
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        return nce


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')if features.is_cudaelse torch.device('cpu'))
        device = torch.device('cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class VAT(nn.Module):
    def __init__(self, model,device):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.model = model
        self.epsilon = 3.5
        self.device = device

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device=self.device)

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m, features_m = self.model(x + d)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m,_ = self.model(x + r_vadv)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss

def gradient_penalty(critic, h_s, h_t,device):
    from torch.autograd import grad
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty




class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward (self, source, target):
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss/(4*d*d)
        return loss



####################### FOR DCAN method ######################################
def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def MMD_reg(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size_source = int(source.size()[0])
    batch_size_target = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size_source):
        s1, s2 = i, (i + 1) % batch_size_source
        t1, t2 = s1 + batch_size_target, s2 + batch_size_target
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size_source + batch_size_target)


##############################################################################
class HoMMD_loss(nn.Module):
    def __init__(self):
        super(HoMMD_loss, self).__init__()

    def forward(self,xs,xt):
        xs = xs - torch.mean(xs, axis=0)
        xt = xt - torch.mean(xt, axis=0)
        xs =  torch.unsqueeze(xs,axis=-1)
        xs = torch.unsqueeze(xs, axis=-1)
        xt = torch.unsqueeze(xt, axis=-1)
        xt = torch.unsqueeze(xt, axis=-1)
        xs_1 = xs.permute(0,2,1,3)
        xs_2 = xs.permute(0, 2, 3, 1)
        xt_1 = xt.permute(0, 2, 1, 3)
        xt_2 = xt.permute(0, 2, 3, 1)
        HR_Xs = xs*xs_1*xs_2   # dim: b*L*L*L
        HR_Xs = torch.mean(HR_Xs,axis=0)   #dim: L*L*L
        HR_Xt = xt * xt_1 * xt_2
        HR_Xt = torch.mean(HR_Xt, axis=0)
        return torch.mean((HR_Xs-HR_Xt)**2)

def domain_contrastive_loss(domains_features, domains_labels, temperature,device):
    # masking for the corresponding class labels.
    anchor_feature = domains_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    labels = domains_labels
    labels= labels.contiguous().view(-1, 1)
    # Generate masking for positive and negative pairs.
    mask = torch.eq(labels, labels.T).float().to(device)
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # create inverted identity matrix with same shape as mask.
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device),
                                0)
    # mask-out self-contrast cases
    mask = mask * logits_mask

    # compute log_prob and remove the diagnal
    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mask_sum = mask.sum(1)
    zeros_idx = torch.where(mask_sum == 0)[0]
    mask_sum[zeros_idx] = 1

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    # loss
    loss = (- 1 * mean_log_prob_pos)
    loss = loss.mean()

    return loss


# class VATLoss(nn.Module):
#
#     def __init__(self, xi=10.0, eps=1.0, ip=1):
#         """VAT loss
#         :param xi: hyperparameter of VAT (default: 10.0)
#         :param eps: hyperparameter of VAT (default: 1.0)
#         :param ip: iteration times of computing adv noise (default: 1)
#         """
#         super(VATLoss, self).__init__()
#         self.xi = xi
#         self.eps = eps
#         self.ip = ip
#
#     def forward(self, model, x):
#         with torch.no_grad():
#             pred, _ = model(x)
#
#         # prepare random unit tensor
#         d = torch.rand(x.shape).sub(0.5).to(x.device)
#         d = _l2_normalize(d)
#
#         with _disable_tracking_bn_stats(model):
#             # calc adversarial direction
#             for _ in range(self.ip):
#                 d.requires_grad_()
#                 pred_hat, _ = model(x + self.xi * d)
#                 adv_distance = F.kl_div(pred_hat, pred, reduction='batchmean')
#                 adv_distance.backward()
#                 d = _l2_normalize(d.grad)
#                 model.zero_grad()
#             # calc LDS
#             r_adv = d * self.eps
#             pred_hat, _ = model(x + r_adv)
#             lds = F.kl_div(pred_hat, pred, reduction='batchmean')
#         return lds