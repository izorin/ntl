import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform

import torch
from torch import utils
from torch.utils.data import Dataset, DataLoader




def download_data(save_path):
    # Download data from GitHub repository
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    os.system(
        "wget -P {} -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01".format(
            save_path
        )
    )
    os.system(
        "wget -P {} -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02".format(
            save_path
        )
    )
    os.system(
        "wget -P {} -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip".format(
            save_path
        )
    )

    # Unzip downloaded data
    file_path = os.path.join(save_path, "data_compress")
    data_1_path = os.path.join(save_path, "data.z01")
    data_2_path = os.path.join(save_path, "data.z02")
    data_z_path = os.path.join(save_path, "data.zip")
    os.system("cat {} {} {} > {}".format(data_1_path, data_2_path, data_z_path, file_path))
    os.system("unzip -n -q {} -d {}".format(file_path, save_path))


def get_dataset(filepath):
    """## Saving "flags" """

    df_raw = pd.read_csv(filepath, index_col=0)
    flags = df_raw.FLAG.copy()

    df_raw.drop(["FLAG"], axis=1, inplace=True)

    """## Sorting"""

    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)
    df_raw = df_raw.T.copy()
    df_raw["FLAG"] = flags
    return df_raw


def get_processed_dataset(filepath):

    df_raw = get_dataset(filepath)
    flags = df_raw["FLAG"]
    df_raw.drop(["FLAG"], axis=1, inplace=True)

    """## Quantile transform"""

    quantile = quantile_transform(
        df_raw.values, n_quantiles=10, random_state=0, copy=True, output_distribution="uniform"
    )
    df__ = pd.DataFrame(data=quantile, columns=df_raw.columns, index=df_raw.index)
    df__["flags"] = flags

    return df__


# torch datasets 
class SGCCDataset(Dataset):
    def __init__(self, path, label=None, scale=None, nan_ratio=1.0, transforms=[], year=None):
        super(SGCCDataset).__init__()
        self.path = path
        self.label = label
        self.scale = scale
        self.nan_ratio = nan_ratio
        self.transforms = transforms
        # loading dataset
        self.data = self._get_dataset()
        self.data = self._filter_by_label(self.data, self.label) # extracting data of only selected class
        self.labels = self.data['FLAG'].to_numpy() # class labels
        self.data = self.data.drop('FLAG', axis=1)
        self._filter_by_nan_ratio()
        self._fill_na_() # filling NaN in consumption
        self.consumers = self.data.reset_index()['CONS_NO'].to_list() # names of consumers

        if year:
            #TODO: slice data to have only selected year
            # transpose raw_data and pick year in index
            pass
        
        self.length = self.data.shape[0]
        self.data = self.data.to_numpy()
        if self.scale:
            self.data = self._scale_data()
            
    def _scale_data(self):
        eps = 10 ** -8
        if self.scale == 'minmax':
            _min = self.data.min(axis=1)
            _max = self.data.max(axis=1)
            self.data = (self.data - _min[:, None]) / (_max[:, None] - _min[:, None] + eps)

        elif self.scale == 'standard':
            mean = self.data.mean(axis=1)
            std = self.data.std(axis=1)
            self.data = (self.data - mean[:, None]) / (std[:, None] + eps)
            
        else:
            print('unknow scaler name')
            ValueError

        return self.data

    def _filter_by_label(self, data, label):
        if label in ('normal', 0):
            data = data[data['FLAG'] == 0]
        elif label in ('anomal', 1):
            data = data[data['FLAG'] == 1]
        else:
            pass

        return data

    def _filter_by_nan_ratio(self):
        days = self.data.shape[1] # length of time series
        consumers_nan_ratio = self.data.isna().sum(axis=1) / days # missing value ratio per consumer
        consumers_nan_ratio = consumers_nan_ratio[consumers_nan_ratio < self.nan_ratio] 
        self.data = self.data.loc[consumers_nan_ratio.index.to_list()]
    
    def _fill_na_(self):
        # filling with zeros
        self.data.fillna(0, inplace=True)

    def _get_dataset(self):
        return get_dataset(self.path)

    def _get_item(self, idx):
        return (self.labels[idx], self.data[idx, :, None].astype(np.float32), self.consumers[idx])

    def __getitem__(self, idx):
        sample = self._get_item(idx)
        
        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def __len__(self):
        return self.length



def sgcc_train_test_split(config):
    normal_dataset = SGCCDataset(path=config.data_path, label=0, scale=config.scale, nan_ratio=config.nan_ratio)
    anomal_dataset = SGCCDataset(path=config.data_path, label=1, scale=config.scale)

    train_data, val_data, test_normal_data = utils.data.random_split(normal_dataset, [len(normal_dataset) - 2*len(anomal_dataset), len(anomal_dataset), len(anomal_dataset)])
    test_data = utils.data.ConcatDataset([test_normal_data, anomal_dataset])
    
    return train_data, val_data, test_data


def define_loaders(config):
    train_data, val_data, test_data = sgcc_train_test_split(config)
    
    train_loader = DataLoader()
    val_loader = DataLoader()
    test_loader = DataLoader()
    
    return train_loader, val_loader, test_loader
    
    
    
class DummyDataset(Dataset):
    def __init__(self, dim, length=1000):
        self.dim = dim
        self.len = length
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return torch.randn([self.dim])
    
    