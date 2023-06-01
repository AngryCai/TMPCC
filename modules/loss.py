import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)

        sim = torch.matmul(z, z.T) / self.temperature  # Dot similarity
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class CrossCorrelationLoss(nn.Module):

    def __init__(self, out_dim, lambd, device):
        super(CrossCorrelationLoss, self).__init__()
        self.lambd = lambd
        self.device = device
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, y_i, y_j):
        batch_size = y_i.size(0)
        c = self.bn(y_i).T @ self.bn(y_j)
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ClusteringLoss(nn.Module):
    def __init__(self, weight_clu_loss=0.01, regularization_coef=0.05):
        super(ClusteringLoss, self).__init__()
        self.kl_criterion = nn.KLDivLoss(reduction='sum')
        self.regularization_coef = regularization_coef
        self.weight_clu_loss = weight_clu_loss

    def forward(self, y_prob, cluster_center=None):
        """
        :param y_prob: prob of embeddings
        :return:
        """
        target_prob = self.target_distribution(y_prob)  # .detach()
        loss = self.kl_criterion(y_prob.log(), target_prob)/y_prob.shape[0]
        reg_loss = 0.
        if cluster_center is not None:  # #  orthogonal regularization on centers: matmul(C, C^T) - I
            # cluster_center = F.normalize(cluster_center)
            x = torch.matmul(cluster_center, cluster_center.t())
            n, m = x.shape
            reg_loss = torch.norm(x - torch.eye(n).to(cluster_center.device)).pow_(2).sum()
            # off_diag = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten().pow(2).sum()
            # loss += 1e-5 * reg_loss
        prob = y_prob.sum(0).view(-1)
        prob /= prob.sum()
        entropy = math.log(prob.size(0)) + (prob * torch.log(prob)).sum()
        loss = self.weight_clu_loss * loss + self.regularization_coef * (entropy + reg_loss)
        # loss += 1e-5 * reg_loss
        return loss

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-8)
        return (weight.t() / torch.sum(weight, 1)).t()


class PretrainLoss(nn.Module):
    """
    pretrain model for n epoch with a contrastive task, e.g., SimCLR/BarlowTwins
    """
    def __init__(self, batch_size, lambda_, device='cpu'):
        super(PretrainLoss, self).__init__()
        self.device = device
        self.criterion = InstanceLoss(batch_size, lambda_, device).to(device)
        # self.criterion = CrossCorrelationLoss(batch_size, lambda_, device).to(device)

    def forward(self, x_1, x_2):
        loss = self.criterion(x_1, x_2)
        return loss


class JointLoss(nn.Module):
    """
    joint train model with a center-based loss plussed with a contrastive loss
    """
    def __init__(self, batch_size, lambda_=0.5, weight_clu=1, regularization_coef=0.05, device='cpu'):
        super(JointLoss, self).__init__()
        self.device = device
        self.weight_clu = weight_clu
        self.regularization_coef = regularization_coef
        self.criterion_contrastive = InstanceLoss(batch_size, lambda_, device).to(device)
        self.clustering_loss = ClusteringLoss(weight_clu, regularization_coef)

    def forward(self, y_1, y_2, cluster_center=None):
        h = torch.cat([y_1, y_2], dim=0)
        loss_con = self.criterion_contrastive(y_1, y_2)
        loss_clu = self.clustering_loss(h, cluster_center)
        loss = loss_con + loss_clu
        return loss, loss_con, loss_clu


