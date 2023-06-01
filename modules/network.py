import torch
import torch.nn as nn
from modules.vit import ViT


class PatchEmbedding(nn.Module):
    """
    transform different modalities into the same dim
    """

    def __init__(self, n_modalities, in_channels, out_channel):
        """
        :param n_modalities: number of modalities
        :param in_channels: tuple of input channels of multiple modalities, or a single modality
        :param out_channel:
        """
        super(PatchEmbedding, self).__init__()
        self.n_modalities = n_modalities
        self.out_channel = out_channel
        self.in_channels = in_channels
        if not isinstance(self.in_channels, tuple):
            self.in_channels = (self.in_channels,)
        self.layers = nn.ModuleList([nn.Conv2d(self.in_channels[i], out_channel, (3, 3)) for i in range(n_modalities)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_channel) for i in range(n_modalities)])

    def forward(self, x):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        x = [bn(layer(x_i)) for x_i, layer, bn in zip(x, self.layers, self.bn)]
        x = torch.cat(x, dim=-1)
        return x


class ContrastiveHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ContrastiveHead, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class ClusteringHead(nn.Module):
    def __init__(self, n_dim, n_class, alpha=1.):
        super(ClusteringHead, self).__init__()
        # Clustering head
        self.alpha = alpha
        # initial_cluster_centers = torch.tensor(torch.randn((n_class, n_dim), dtype=torch.float, requires_grad=True))
        self.cluster_centers = nn.Parameter(torch.Tensor(n_class, n_dim), requires_grad=True)
        # torch.nn.init.orthogonal_(self.cluster_centers.data, gain=1)
        torch.nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        """
        :param x: n_batch * n-dim
        :return:
        """
        pred_prob = self.get_cluster_prob(x)
        return pred_prob

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class Net(nn.Module):
    def __init__(self, n_modalities, in_channels, in_patch_size, common_channel, n_class, dim_emebeding):
        super(Net, self).__init__()
        self.embedding_layer = PatchEmbedding(n_modalities, in_channels, common_channel)
        self.vit = ViT(image_size=(in_patch_size[0]-2, (in_patch_size[1]-2) * 2),  # use 3*3 kernel in embedding layer #(5, 10),
                       # image_size=(in_patch_size[0], in_patch_size[1] * 2),
                       patch_size=1,
                       # num_classes=n_class,
                       dim=512,
                       depth=4,
                       heads=8,
                       mlp_dim=1024,
                       pool='mean',
                       channels=common_channel,
                       dim_head=64,
                       dropout=0.1,
                       emb_dropout=0.1
                       )
        self.clustering_head = ClusteringHead(dim_emebeding, n_class, alpha=1) ## ContrastiveHead(512, 128)

    def forward(self, x_1, x_2):
        """
        :param x_1, x_2: tuple of modalities, e.g., [aug_1, aug_2]-->
        ([img_rgb, img_hsi, img_sar], [img_rgb, img_hsi, img_sar])
        :return:
        """
        x_1 = self.vit(self.embedding_layer(x_1))  # # concatenated modalities: [batch, n_channel, width, 2*height]
        x_2 = self.vit(self.embedding_layer(x_2))

        y_1 = self.clustering_head(x_1)
        y_2 = self.clustering_head(x_2)

        return y_1, y_2

    def forward_embedding(self, x):
        # h = self.clustering_head(self.vit(self.embedding_layer(x)))
        h = self.vit(self.embedding_layer(x))
        return h

    def forward_cluster(self, x, return_h=False):
        """
        :param x: tuple of modalities, e.g., (img_rgb, img_hsi, img_sar)
        :return:
        """
        h = self.vit(self.embedding_layer(x))
        pred = self.clustering_head(h)
        labels = torch.argmax(pred, dim=1)
        if return_h:
            return labels, h
        return labels
