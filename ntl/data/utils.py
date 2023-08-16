
import os
import numpy as np
import pandas as pd
import fire


from sklearn.preprocessing import quantile_transform
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold


def download_data(filepath):
    # Download data from GitHub repository
    os.system(f'wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01 -O {filepath}')
    os.system(f'wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02 -O {filepath}')
    os.system(f'wget -nc -q https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip -O {filepath}')

    # Unzip downloaded data
    os.system('cat data.z01 data.z02 data.zip > data_compress.zip')
    os.system('unzip -n -q data_compress')


def get_dataset(filepath):
    """## Saving "flags" """
    df_raw = pd.read_csv(filepath,index_col=0)
    flags = df_raw.FLAG.copy()
 
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Sorting"""
    df_raw = df_raw.T.copy()
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw.sort_index(inplace=True, axis=0)
    df_raw = df_raw.T.copy()
    df_raw['FLAG'] = flags
    return df_raw


def get_processed_dataset(filepath):
    df_raw = get_dataset(filepath)
    flags = df_raw['FLAG']
    df_raw.drop(['FLAG'], axis=1, inplace=True)

    """## Quantile transform"""
    quantile = quantile_transform(df_raw.values, n_quantiles=10, random_state=0, copy=True, output_distribution='uniform')
    df__ = pd.DataFrame(data=quantile, columns=df_raw.columns, index=df_raw.index)
    df__['flags'] = flags

    return df__.iloc[:, 5:]


def get_data(filepath):

    if not os.path.exists(os.path.join(filepath, 'data.csv')):
        if not os.path.exists(os.path.join(filepath, 'data.zip')):
            download_data(filepath)

    data = get_processed_dataset(filepath)
    return data



if __name__ == '__main__':
    fire.Fire(get_data)
