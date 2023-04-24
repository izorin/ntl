import sys
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import quantile_transform


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


