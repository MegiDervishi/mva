"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    
    ##################
    A = A + sp.identity(A.shape[0])
    diag = A @ np.ones(A.shape[0])
    inv_diag = np.power(diag, -1)
    D_inv = sp.diags(inv_diag)
    A_normalized = D_inv @ A
    ##################
    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3
    
    idx = adj._indices()
    
    A_hat_1 = torch.einsum( 'ij, ij -> i', z[idx[0, :], :], z[idx[1, :], :] ) 
    A_tilde_1 = torch.ones(idx.size(1)).to(device)
    
    idx_0 = torch.randint(z.size(0), idx.size())
    
    A_hat_0 = torch.einsum( 'ij, ij -> i', z[idx_0[0, :], :], z[idx_0[1, :], :] ) 
    A_tilde_0 = torch.zeros(idx.size(1)).to(device)
    
    A_hat = torch.cat([A_hat_1, A_hat_0])
    A_tilde = torch.cat([A_tilde_1, A_tilde_0])
    
    loss = mse_loss(A_hat, A_tilde)
    return loss
