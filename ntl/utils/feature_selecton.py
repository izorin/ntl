#%%
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tsfresh as tf
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import feature_selection as fs

from yellowbrick.features import pca,  pca_decomposition, manifold_embedding
from yellowbrick import features as feat

import sys
sys.path.append('/Users/ivan_zorin/Documents/AIRI/code/twino/')

from research_src.data import *



#%% func declaration 
def get_tsf_features(df, path, is_select_important=True, is_save=True):
    labels = df['FLAG'] # DataFrame with labels 

    if os.path.exists(path):
        features = pd.read_csv(path)
    
    else:
        # flat table of data : |cons_no | date | consumption | 
        data = df.drop('FLAG', axis=1).reset_index().melt(id_vars=['CONS_NO'], var_name='date', value_name='cons').fillna(0)
        
        # TODO hardcoded year 
        features = extract_features(data[data.date.dt.year == 2016], column_id='CONS_NO', column_sort='date', column_value='cons')
        impute(features)
        features.reset_index(inplace=True)
        features.rename({'index' : 'consumer'}, axis=1, inplace=True)
        if is_save:
            features.to_csv(path)
    
    # features = pd.merge(features, labels, left_on='consumer', right_on='CONS_NO')
    if is_select_important:
        features = fs.selection.select_features(features.set_index('consumer'), labels, ml_task='classification')

    return features, labels



#%%
# get data 
def main():
    PATH = '/Users/ivan_zorin/Documents/AIRI/data/sgcc/data.csv'
    df = get_dataset(PATH) # raw data

    features_path = '/Users/ivan_zorin/Documents/AIRI/data/sgcc/features.csv'
    features, labels = get_tsf_features(df, features_path, is_select_important=True)

    print(features.head())
    print(labels.head())
# %%

if __name__ == '__main__':
    main()