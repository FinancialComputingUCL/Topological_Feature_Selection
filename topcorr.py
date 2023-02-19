import collections

import pandas as pd
import numpy as np

from tmfg_core import *

import networkx as nx

from sklearn.preprocessing import MinMaxScaler

def tmfg(data, method, dataset_name, correlation_type, alpha, step, threshold_mean=True):
    
    if method == 'pearson':
        if step == 'cv':
            corr = pd.DataFrame(data).corr(method='pearson').to_numpy()
        elif step == 'test':
            corr = pd.read_csv(f'{method}_{dataset_name}.csv').to_numpy()

    elif method == 'spearman':
        if step == 'cv':
            corr = pd.DataFrame(data).corr(method='spearman').to_numpy()
        elif step == 'test':
            corr = pd.read_csv(f'{method}_{dataset_name}.csv').to_numpy()
    else:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        std = scaled_data.std(axis=0).reshape([-1,1])
        sigma_ij = np.maximum(std, std.transpose())

        if step == 'cv':
            corr_ij = 1 - np.abs(pd.DataFrame(data).corr(method='spearman').to_numpy())
        elif step == 'test':
            corr_ij = 1 - np.abs(pd.read_csv(f'spearman_{dataset_name}.csv').to_numpy())

        corr = alpha * sigma_ij + (1-alpha) * corr_ij
        
    corr = np.nan_to_num(corr, 0)
    
    p = corr.shape[0]

    weight_corr = corr
    
    if correlation_type == 'square' and method != 'energy':
        weight_corr = np.square(corr)

    if threshold_mean:
        weight_corr[weight_corr < weight_corr.mean()] = 0

    tmfg = TMFG(pd.DataFrame(weight_corr))
    cliques, seps, JS = tmfg.compute_TMFG()
    return nx.from_numpy_matrix(JS)