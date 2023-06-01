import torch
import torchvision
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn import random_projection


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, img):
        prob = np.random.random_sample()
        if prob < 0.5:
            img = np.array(img)
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), sigma)
            img = torch.from_numpy(img).float()
        return img


class Transforms:
    def __init__(self, size, mean=None, std=None, blur=False):
        self.train_transform = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomChoice([GaussianBlur(3),
                                                 MaskPixels(p=0.2),
                                                 MaskBands(p=0.2)
                                                ], p=[0.4, 0.5, 0.1]),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=3))
        # self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            # torchvision.transforms.Resize(size=(size, size)),
            # MaskBands(),
            # RandomProjectionBands(n_band=200)
            # torchvision.transforms.ToTensor(),
            # MaskBands(p=0.2),
            # RandomProjectionBands(n_band=32),
            # PermuteBands(10)
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class GroupPermuteBands(object):
    """
    shuffle bands into n_groups
    """
    def __init__(self, n_group=3):
        self.n_group = n_group

    def __call__(self, img):
        n_channel = img.size(0)
        n_group_band = int(np.ceil(n_channel / self.n_group))
        for i in range(self.n_group):
            start = i * n_group_band
            end = start + n_group_band
            if end >= n_channel:
                indx = np.arange(start, n_channel)
                indx_ = np.arange(start, n_channel)
            else:
                indx = np.arange(start, end)
                indx_ = np.arange(start, end)
            np.random.shuffle(indx)
            img[indx_] = img[indx]
        # indx_selected = indx[:n_shuffle]
        # select_mask = np.zeros((n_channel, 1, 1))
        # select_mask[indx_selected] = 1
        # img_shuffled = img[indx]

        return img


class MaskPixels(object):
    def __init__(self, p=0.5):
        """
        :param p:  every pixel will be masked  with a probability of p
        """
        self.p = 1 - p

    def __call__(self, img):
        n_band, h, w = img.shape
        mask = np.random.binomial(1, self.p, size=(h, w))
        mask = torch.from_numpy(mask).float()
        mask = mask.expand((n_band, h, w))
        img = mask * img
        return img


class MaskBands(object):

    def __init__(self, p=0.5):
        """

        :param p: a band will be masked with probability of p
        """
        self.p = 1. - p

    def __call__(self, img):
        # indx = np.arange(img.shape[0])
        # indx_selected = np.random.choice(indx, self.n_band, replace=False)
        # img = img[indx_selected]

        prob = np.random.binomial(1, self.p, img.shape[0])
        prob = np.reshape(prob, (img.shape[0], 1, 1))
        prob = torch.from_numpy(prob).float()
        # img = img[np.where(prob == 1)]
        img = img * prob
        return img


class RandomProjectionBands(object):

    def __init__(self, n_band=None):
        """
        :param n_band: project to n_band
        """
        self.n_band = n_band

    def __call__(self, img):
        # # n_band * w * h
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        n_band, h, w = img.shape
        if self.n_band is None:
            # self.n_band = np.random.randint(3, n_band//2)
            transformer = random_projection.SparseRandomProjection(n_components='auto')
        else:
            transformer = random_projection.SparseRandomProjection(n_components=self.n_band)
        img_ = img.transpose((1, 2, 0))
        x_2d = img_.reshape((-1, n_band))
        x_2d_ = transformer.fit_transform(x_2d)
        img_new = x_2d_.reshape((h, w, -1)).transpose(2, 0, 1)
        img_new = torch.from_numpy(img_new).float()
        return img_new


class ShufflePixel(object):

    def __init__(self):
        pass

    def __call__(self, img):
        n_band, h, w = img.shape
        img_ = img.view(n_band, -1)
        img_ = img_[torch.randperm(n_band)]
        img = img_.view(n_band, h, w)
        return img

