import torch
from torch.utils.data import Dataset
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, file_path, set_type):
        with h5py.File(file_path, 'r') as f:
            X_set = f[f'X_{set_type}']
            y_set = f[f'y_{set_type}']

            self.features = []

            if 'X_train' in f:
                if 'features' in f['X_train'].attrs:
                    self.features = f['X_train'].attrs['features']

            self.x_features = X_set.shape[-1]
            self.x_len = X_set.shape[1]
            self.y_len = y_set.shape[1]
            self.data_len = X_set.shape[0]
            self.samples = self.data_len

            self.X = X_set[:]
            self.y = y_set[:]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.X[idx]
        target = self.y[idx]

        return torch.tensor(data), torch.tensor(target)