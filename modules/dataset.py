import torch
from torch.utils.data import Dataset, DataLoader
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np


class MultiModalDataset(Dataset):

    def __init__(self, gt_path, *src_path, patch_size=(7, 7), transform=None, is_labeled=True):
        self.transform = transform
        p = Processor()
        n_modality = len(src_path)
        modality_list = []
        in_channels = []
        for i in range(n_modality):
            img, gt = p.prepare_data(src_path[i], gt_path)
            x_patches, y_ = p.get_HSI_patches_rw(img, gt, (patch_size[0], patch_size[1]), is_indix=False, is_labeled=is_labeled)
            n_samples, n_row, n_col, n_channel = x_patches.shape
            scaler = StandardScaler()
            batch_size = 5000
            # # using incremental / batch for very large data
            for start_id in range(0, x_patches.shape[0], batch_size):
                n_batch = x_patches[start_id: start_id+batch_size].shape[0]
                scaler.partial_fit(x_patches[start_id: start_id+batch_size].reshape(n_batch, -1))
            for start_id in range(0, x_patches.shape[0], batch_size):
                shape = x_patches[start_id: start_id+batch_size].shape
                x_temp = x_patches[start_id: start_id+batch_size].reshape(shape[0], -1)
                x_patches[start_id: start_id+batch_size] = scaler.transform(x_temp).reshape(shape)
            x_patches = np.transpose(x_patches, axes=(0, 3, 1, 2))
            x_tensor = torch.from_numpy(x_patches).type(torch.FloatTensor)
            modality_list.append(x_tensor)
            in_channels.append(n_channel)
        y = p.standardize_label(y_)
        self.gt_shape = gt.shape
        self.data_size = len(y)
        if is_labeled:
            self.n_classes = np.unique(y).shape[0]
        else:
            self.n_classes = np.unique(y).shape[0] - 1  # remove background
        self.y_tensor = torch.from_numpy(y).type(torch.LongTensor)
        self.modality_list = tuple(modality_list)
        self.n_modality = n_modality
        self.in_channels = tuple(in_channels)

    def __getitem__(self, idx):
        x_list = []
        for i in range(self.n_modality):
            x = self.modality_list[i][idx]
            if self.transform is not None:
                x_1, x_2 = self.transform(x)  # # conduct transformation on a single modality
                x_list.append(x_1)
                x_list.append(x_2)
            else:
                x_list.append(x)
        if self.n_modality >= 2 and len(x_list) > 2:  # # when modality >= 2, i.e., 4 augs
            x_list = (x_list[0::2], x_list[1::2])
        if self.n_modality == 1 and len(x_list) == 2:
            x_list = ([x_list[0]], [x_list[1]])
        y = self.y_tensor[idx]
        return x_list, y

    def __len__(self):
        return self.data_size
