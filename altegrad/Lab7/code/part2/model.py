"""
Graph-based Recommendations - ALTEGRAD - Jan 2022
"""

import math

import torch
import torch.nn as nn
import ipdb

class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MessagePassing, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, adj, x):
        
        ############## Task 9
    
        ##################
        x = self.fc(x)
        output = torch.mm(adj,x)
        ##################
        
        return output


class SR_GNN(nn.Module):
    def __init__(self, n_items, hidden_dim, dropout, device):
        super(SR_GNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(n_items, hidden_dim)
        self.mp1 = MessagePassing(hidden_dim, hidden_dim)
        self.mp2 = MessagePassing(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, items, last_item, idx):
        
        ############## Task 10
        #ipdb.set_trace()
        ##################
        x = self.embedding(items)
        h = self.relu(self.mp1(adj, x))
        h = self.dropout(h)
        z = self.mp2(adj, h)
        z = self.dropout(z)
        ##################
        
        sl = z[last_item]
        
        q1 = self.fc1(sl)
        q1 = torch.index_select(q1, 0, idx)
        q2 = self.fc2(z)
        alpha = self.fc3(torch.sigmoid(q1 + q2))
        z = alpha*z
        idx = idx.unsqueeze(1).repeat(1, z.size(1))
        out = torch.zeros(last_item.size(0), z.size(1)).to(self.device)
        sg = out.scatter_add_(0, idx, z)
        sg = self.dropout(sg)
        
        ##################
        s = torch.cat((sl, sg), dim=1)
        sh = self.fc4(s)
        out = torch.mm(self.embedding.weight, torch.transpose(sh, 0, 1))
        out = nn.functional.softmax(out, dim=0) 
        ##################
        
        return torch.transpose(out, 0, 1)