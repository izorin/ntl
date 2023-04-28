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
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import yaml
from types import SimpleNamespace


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


def reduce_embeddings(embeddings, method, labels=None, plot=True, title=''):
    
    if method.upper() == 'PCA':
        method = PCA(2)
        coord = method.fit_transform(embeddings)
    
    elif method.upper() == 'TSNE':
        method = TSNE(2)
        coord = method.fit_transform(embeddings)
    else:
        print(f'unknow dim reductin method: {method}')
        ValueError
        
    if plot:
        plt.figure()
        sns.scatterplot(x=coord[:, 0], y=coord[:, 1], color='b', alpha=0.2)
        plt.legend()
        plt.title(title)
        plt.show()
        
    return coord


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    config = SimpleNamespace(**config)
    return config
    
    