import numpy as np
from sklearn import preprocessing
from torch.utils.data import TensorDataset, random_split
import torch
import ipdb

def load2torch(data_num, data_dir = None):
    if not data_dir:
        X = np.load(f'.\\save\\train\\X_train{data_num}.npy', allow_pickle=True)
    else:
        X = np.load(f'{data_dir}\\X_train{data_num}.npy', allow_pickle=True)
    for i in range(len(X)):
        X[i][-3] = X[i][-3][0, 0]

    n_samples, d_in = X.shape
    if not data_dir:
        y = np.load(f'.\\save\\train\\y_train{data_num}.npy', allow_pickle=True)
    else:
        y = np.load(f'{data_dir}\\y_train{data_num}.npy', allow_pickle=True)
    X = np.clip(X, -3e38, 3e38)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    X, y = X.float(), y.type(torch.LongTensor)
    dataset = TensorDataset(X, y)
    train_samples = 90 * n_samples//100
    val_samples = n_samples - train_samples
    train_dataset, val_dataset = random_split(dataset, [train_samples, val_samples])
    return train_dataset, val_dataset, d_in

def test2torch(data_num):
    X = np.load(f'.\\save\\test\\X_test{data_num}.npy', allow_pickle=True)
    for i in range(len(X)):
        X[i][-3] = X[i][-3][0, 0]

    X = np.clip(X, -3e38, 3e38)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.from_numpy(X).float()
    return X