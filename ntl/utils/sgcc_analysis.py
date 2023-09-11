#%%
from sgcc_load_data import get_data, get_processed_dataset, get_dataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as st
import statsmodels.api as sta
from sklearn.decomposition import PCA

np.random.seed(42)
#%%
filepath = '/Users/ivan_zorin/Documents/AIRI/data/sgcc/data.csv'
# raw_data = get_processed_dataset(filepath)
raw_data = get_dataset(filepath)

labels = raw_data['FLAG']
data = (raw_data
            .copy()
            .drop(axis=1, labels='FLAG')
            .transpose()
)
data = data.transpose()
# data.reset_index(inplace=True)
#%% NaN heatmap
plt.figure()
sns.heatmap(data.isna().to_numpy(), yticklabels=False, xticklabels=False)
plt.ylabel('consumers')
plt.xlabel('time')
plt.title('Missing values heatmap')
plt.show()

#%% histogram of NaNs per customer
nans = data.isna().sum(axis=1)
nans_pct = nans / data.shape[1]
nans_pct = pd.concat([nans_pct, labels], axis=1)

#%%
# plt.figure()
# sns.histplot(nans_pct)
# plt.title('total NaN hist')
# plt.show()
#%% hist of nans of normal and thieves
plt.figure()
sns.displot(data=nans_pct, x=0, hue='FLAG', kind='hist', multiple='stack', stat='probability', common_norm=False, kde=True, legend=False)
plt.legend(['thieves', 'norm'])
plt.xlabel('')
plt.title('Distribution of NaNs of two classes')
plt.show()

#%% separate plots of hists of norm and thieves
nans_norm = data.loc[labels[labels == 0].index].isna().sum(axis=1) / data.shape[1]
nans_thieves = data.loc[labels[labels == 1].index].isna().sum(axis=1) / data.shape[1]

fig, axs = plt.subplots(1,1)
sns.histplot(nans_norm, ax=axs, color='blue')
sns.histplot(nans_thieves, ax=axs, color='y')
# sns.histplot(nans_thieves, ax=axs[1])
plt.show()


# %% consumption 
consumption = (raw_data
                .reset_index()
                .melt(id_vars=['CONS_NO', 'FLAG'], var_name='date', value_name='cons')
)
# %% consumption plot and  random normal and thief 
def plot_consumer(consumer, title_name, color=None):
    fig, axs = plt.subplots(2,1)
    # line plot
    line = sns.lineplot(consumption[consumption.CONS_NO == consumer], x='date', y='cons', hue='CONS_NO', ax=axs[0], legend=False, palette=[color])
    line.set_xticklabels(line.get_xticklabels(), rotation=45)
    line.set_ylabel('consumption')
    # hist 
    sns.histplot(data=consumption[consumption.CONS_NO == consumer], x='cons', kde=True, ax=axs[1], color=color)
    plt.suptitle(f'{title_name} consumer')
    fig.tight_layout()
    plt.show()

norm_consumer = np.random.choice(consumption[consumption['FLAG'] == 0].CONS_NO)
thief_consumer = np.random.choice(consumption[consumption['FLAG'] == 1].CONS_NO)

# cherry picked consumers for plots 
norm_consumer = '0123C59D843937C6DAC0E978E5E05B43'
thief_consumer = 'EA6A61D91EE6180AFFDE6E650FE80E9A'

plot_consumer(norm_consumer, 'Normal', 'b')
plot_consumer(thief_consumer, 'Thief', 'r')

#%% ACF & pACF
norm_consumer = np.random.choice(consumption[consumption['FLAG'] == 0].CONS_NO)
x = consumption[consumption.CONS_NO == norm_consumer].cons.fillna(0).to_numpy()

# pacf = st.tsa.stattools.acf(x)
st.graphics.tsaplots.plot_acf(x);
st.graphics.tsaplots.plot_pacf(x);

#%%
y = consumption[consumption.CONS_NO == norm_consumer][['date', 'cons']].fillna(0)
y['date'] = pd.to_datetime(y.date)
st.graphics.tsaplots.month_plot(y.cons.to_numpy(), dates=y.date);


# %%

#%% pointwise average consumption 
mean_all = data.mean(axis=0)
mean_norm = raw_data[raw_data['FLAG'] == 0].drop(axis=1, labels='FLAG').mean(axis=0)
mean_thief = raw_data[raw_data['FLAG'] == 1].drop(axis=1, labels='FLAG').mean(axis=0)

#%%
plt.figure()
# sns.lineplot(mean_all, label='all')
sns.lineplot(mean_norm, label='norm')
sns.lineplot(mean_thief, label='thief')
plt.xticks(rotation=45)
plt.show()

#%% flatting data (cons_no | date | flag | consumption)
data_flat = raw_data.reset_index().melt(id_vars=['CONS_NO', 'FLAG'], var_name='date', value_name='cons')
#%%
plt.figure()
avg_plot = sns.lineplot(data=data_flat, x='date', y='cons', hue='FLAG', estimator='mean', errorbar='se')

plt.title('Pointwise average with standard error interval')
avg_plot.set_ylabel('consumption')
avg_plot.set_xticklabels(avg_plot.get_xticklabels(), rotation=45)
avg_plot.legend(labels=['normal', '_none_', 'thieves', '_none_'])
plt.show()

#%% USELESS
# plt.figure()
# sns.boxplot(data=data_flat, x='date', y='cons', hue='FLAG', orient='v')
# plt.show()

# plt.figure()
# sns.scatterplot(data=data_flat, x='date', y='cons', hue='FLAG', alpha=0.3)
# plt.show()

#%% trying TimeCluster
# rolling window with stride
x = consumption[consumption.CONS_NO == norm_consumer].cons.fillna(0).to_numpy()
#%%
window_len = 50
stride = 1
Z  = [x[i: i + window_len] for i in range(0, len(x) - window_len, stride)]
Z = np.stack(Z)

x_red = PCA(2).fit_transform(Z)

plt.figure()
plt.plot(x_red[:,0], x_red[:,1], '-o')
plt.show()

#%% hist of customers' mean and median over time
stats = (data_flat
            .groupby(by=['CONS_NO']).cons
            .agg([np.mean, np.median])
            .reset_index()
)

stats = stats.merge(labels.reset_index(), on='CONS_NO')
# %% 
plt.figure()
sns.histplot(data=stats, x='mean', bins=20, hue='FLAG', multiple='stack', common_norm=False, stat='probability', kde=False)
plt.title("Distribution of consumers' mean consumption")
plt.legend(['thieves', 'normal'])
plt.show()

plt.figure()
sns.histplot(data=stats, x='median', bins=20, hue='FLAG', multiple='stack', common_norm=False, stat='probability', kde=False)
plt.title("Distribution of consumers' median consumption")
plt.legend(['thieves', 'normal'])
plt.show()

#%%
plt.figure()
sns.swarmplot(data=stats, x='mean', hue='FLAG')
plt.show()

#%% ARIMA
norm_consumer = np.random.choice(consumption[consumption['FLAG'] == 0].CONS_NO)
x = consumption[consumption.CONS_NO == norm_consumer].cons.fillna(0).to_numpy()
#%%
arima = st.tsa.arima.model.ARIMA(x[:-100], order=(5,1,0), )
arima_res = arima.fit()
# %%
plt.figure()
sns.histplot(arima_res.resid)
plt.show()

#%%
arima_res.summary()