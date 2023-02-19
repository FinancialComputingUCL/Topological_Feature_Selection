from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

def inf_fs(data, dataset_name, alpha=0.5, factor=0.9, step='cv'):

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    std = scaled_data.std(axis=0).reshape([-1,1])
    sigma_ij = np.maximum(std, std.transpose())

    if step == 'cv':
        corr_matrix = pd.DataFrame(data).corr(method='spearman').to_numpy()
    elif step == 'test':
        corr_matrix = pd.read_csv(f'spearman_{dataset_name}.csv').to_numpy()

    corr_matrix = np.nan_to_num(corr_matrix, 0)
    corr_ij = 1 - np.abs(corr_matrix)
    A = alpha * sigma_ij + (1-alpha) * corr_ij
    A = np.nan_to_num(A, 0)

    # letting paths tend to infinite
    r = factor/np.max(np.abs(np.linalg.eigvals(A)))
    I = np.eye(A.shape[0])
    S = np.linalg.inv(I-r*A)-I

    energy = np.sum(S, axis=0)
    rank = np.argsort(energy)[::-1]
    return rank, np.sort(energy)[::-1]

def select_inf_fs(data, n_features_to_keep, dataset_name, alpha=0.5, factor=0.9, step='train'):
    rank, score = inf_fs(data, dataset_name, alpha, factor, step)
    rank_n = rank[:n_features_to_keep]
    return np.take(data, rank_n, axis=1), rank_n