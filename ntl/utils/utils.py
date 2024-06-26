import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    auc,
    precision_score,
    cohen_kappa_score,
    confusion_matrix,
)

import random

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score

import yaml
from types import SimpleNamespace

import torch

import wandb
from datetime import datetime

WINDOW_SIZE = 365
def best_window(df, window_size=365):
    columns_nans = df.isna().sum().values
    min_idx = 0
    min_nans = np.sum(columns_nans[:window_size])
    for i in range(len(columns_nans) - 365):
        window_nans = np.sum(columns_nans[i : i + window_size])
        if window_nans < min_nans:
            min_idx = i
            min_nans = window_nans

    return min_idx


def perform_metrics(y_train, y_test, y_pred_train, y_proba_train, y_pred_test, y_proba_test, show=True):
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    roc_auc_train = roc_auc_score(y_train, y_proba_train)
    roc_auc_test = roc_auc_score(y_test, y_proba_test)

    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)

    ck_train = cohen_kappa_score(y_train, y_pred_train)
    ck_test = cohen_kappa_score(y_test, y_pred_test)

    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    if show:
        print("F1 score:            train ---> {:.3}           test ---> {:.3}".format(f1_train, f1_test))
        print(
            "Precision score:     train ---> {:.3}           test ---> {:.3}".format(precision_train, precision_test)
        )
        print("ROC AUC score:       train ---> {:.3}           test ---> {:.3}".format(roc_auc_train, roc_auc_test))
        print("Cohen’s kappa score: train ---> {:.3}           test ---> {:.3}".format(ck_train, ck_test))

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].set_title("ROC - train")
        axs[0].plot(fpr_train, tpr_train, "b", label="AUC = %0.4f" % roc_auc_train)
        axs[0].legend(loc="lower right")
        axs[0].plot([0, 1], [0, 1], "r--")
        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel("True Positive Rate")
        axs[0].set_xlabel("False Positive Rate")

        axs[1].set_title("ROC - test")
        axs[1].plot(fpr_test, tpr_test, "b", label="AUC = %0.4f" % roc_auc_test)
        axs[1].legend(loc="lower right")
        axs[1].plot([0, 1], [0, 1], "r--")
        axs[1].set_xlim(0, 1)
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel("True Positive Rate")
        axs[1].set_xlabel("False Positive Rate")
        plt.show()

    return [
        f1_train,
        f1_test,
        roc_auc_train,
        roc_auc_test,
        precision_train,
        precision_test,
        ck_train,
        ck_test,
        fpr_train,
        fpr_test,
        tpr_train,
        tpr_test,
        threshold_train,
        threshold_test,
        roc_auc_train,
        roc_auc_test,
    ]


def get_conf_matrix(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    print("TN = {}".format(tn))
    print("FP = {}".format(fp))
    print("FN = {}".format(fn))
    print("TP = {}".format(tp))
    return (tn, fp, fn, tp)


def reduce_embeddings(embeddings, method, labels=None, output='plot', title=''):
    
    if method.upper() == 'PCA':
        method = PCA(2)
        coord = method.fit_transform(embeddings)
    
    elif method.upper() == 'TSNE':
        method = TSNE(2)
        coord = method.fit_transform(embeddings)
    else:
        print(f'unknown dim reductin method: {method}')
        ValueError
        
    
    fig = plt.figure()
    sns.scatterplot(x=coord[:, 0], y=coord[:, 1], color='b', alpha=0.2)
    plt.legend()
    plt.title(title)
    if output == 'plot':
        fig.show()
        return coord
        
    if output == 'logger':
        return fig
    
    else:
        print(f'unknown output type ${output}')
        raise ValueError


def load_config(config_file, pathes_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    with open(pathes_file, 'r') as f:
        pathes = yaml.safe_load(f)
    
    for key in pathes.keys():
        config[key] = pathes[key]
        
    config['logger']['save_dir'] = config['save_dir']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
        
    config = SimpleNamespace(**config)
    
    return config
    


def reduce_embed_dim(embs, pca_dim=2):
    embs = PCA(pca_dim).fit_transform(embs)
    if pca_dim > 2:
        embs = TSNE(2).fit_transform(embs)
        
    return embs  

def plot_embeddings(embs, labels=None, title='', log=False, pyplot=False):
    names = {0: 'norm', 1: 'bad'}
    labels = [names[label] for label in labels]
    fig = px.scatter(x=embs[:, 0], y=embs[:, 1], color=labels)
    
    if pyplot:
        fig_plt = plt.figure()
        sns.scatterplot(x=embs[:, 0], y=embs[:, 1], hue=labels)
        return (fig, fig_plt)
    
    return fig


def plot_prediction(x, x_hat, step_name='', pyplot=False):
    fig = go.Figure()
    
    t = np.arange(len(x))
    fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='x', line=dict(color='rgba(0, 0, 255, 0.5)')))
    fig.add_trace(go.Scatter(x=t, y=x_hat, mode='lines', name='x_hat', line=dict(color='rgba(255, 0, 0, 0.7)', dash='dash')))
    fig.update_layout(xaxis_title='i', yaxis_title='x') # title=f'{step_name} GT and prediction'

    if pyplot:
        fig_pyplot = plt.figure()
        sns.lineplot(x=t, y=x, name='x', color='blue', )
        sns.lineplot(x=t, y=x_hat, name='x_hat', color='red')
        return (fig, fig_pyplot)

    return fig


# wandb logger
def log_predicted_signals(x, x_hat, step_name=''):
        
    line_x = wandb.plot.line_series(
        xs = range(x.shape[1]),
        ys=[x, x_hat],
        keys=['x', 'x_hat'],
        title='GT and prediction',
        xname=''
    )
    wandb.log({f'{step_name}/x_prediction': line_x})
    

def compute_roc_auc(scores, labels, pyplot=False):
    # TODO log roc_curve raw data or scores and labels
    
    fpr, tpr, thresh = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    diagonal = np.linspace(0, 1, len(fpr))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=diagonal, y=diagonal, mode='lines', line=dict(color='red', dash='dash')))
    
    fig.update_layout(xaxis_title='FPR', yaxis_title='TPR', showlegend=False)
    fig.add_annotation(x=0.9, y=0.1, text=f'<b>ROC-AUC: {np.round(auc,2)}<b>', showarrow=False )
    
    if pyplot:
        fig_pyplot = plt.figure()
        # sns.lineplot(x=fpr, y=tpr, c='blue')
        # sns.lineplot(x=diagonal, y=diagonal, c='red')
        plt.plot(fpr, tpr, c='blue')
        plt.plot(diagonal, diagonal, c='red')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC-AUC: {np.round(auc,2)}')

        return (fig, fig_pyplot), (fpr, tpr, auc)
    
    
    return fig, (fpr, tpr, auc)

def rec_error_hist(losses, labels, pyplot=False):
    # TODO add edges to the bars 
    class_names = {0: 'norm', 1: 'bad'}
    fig = go.Figure()
    
    classes = set(labels)
    for class_ in classes:
        fig.add_trace(go.Histogram(x=losses[labels == class_], name=class_names[class_]))

    fig.update_layout(barmode='overlay', xaxis_title='reconstruction_error', yaxis_title='count')
    fig.update_traces(opacity=0.75)
    
    if pyplot:
        fig_pyplot = plt.figure()
        for class_ in classes:
            sns.histplot(x=losses[labels == class_], name=class_names[class_])
            
        plt.xlabel('reconstruction error')
        
        return fig, fig_pyplot
    
    return fig


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    # os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def conv2d_shape(hin, pad, dilation, kernel, stride):
    """
    computes output shapes of Conv2d with given parameters 
    
    ```
    conv2d = partial(conv2d_shape, **{'pad': 0 ,'dilation': 1, 'kernel': 3 , 'stride': 1})

    channels = [1, 4, 16, 32, 64]
    hin = [16, 16]
    for channel in channels:
        print(hin)
        hin = [np.floor(conv2d(hin[0])), np.floor(conv2d(hin[1]))]
    ```
    """
    
    hout = (hin + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1 
    return hout

def convtraspose2d_shape(hin, pad, dilation, kernel, stride, output_pad):
    """
    computes output shapes of ConvTraspose2d with given parameters 
    
    ```
    convtranspose2d = partial(convtraspose2d_shape, **{'pad': 0, 'dilation': 1, 'kernel': 3, 'stride': 1, 'output_pad': 0})

    channels = [1, 4, 16, 32, 64]
    hin = [16, 16]

    for channel in channels:
        print(hin)
        hin = [convtranspose2d(hin[0]), convtranspose2d(hin[1])]
    print(hin)
    ```
    """
    hout = (hin - 1) * stride - 2 * pad + dilation * (kernel - 1) + output_pad + 1
    return hout

    
def get_date():
    return datetime.now().strftime('%y_%m_%d_%H_%M')
    