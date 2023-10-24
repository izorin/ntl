import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_auc(labels, scores, n_components=''):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    t = np.linspace(0, 1, fpr.shape[0])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot(t, t, 'r--', alpha=0.4)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC-AUC of GMM({n_components})')
    plt.text(x=0.7, y=0, s=f'AUC: {np.round(auc, 4)}', fontdict={'size': 15})
    plt.show()
        
        
def fit_and_test_gmm(train, test, labels, **gmm_kwargs):
    gmm = GaussianMixture(**gmm_kwargs)
    gmm.fit(train)
    test_scores = gmm.score_samples(test)
    plot_roc_auc(labels, test_scores, n_components=gmm_kwargs['n_components'])