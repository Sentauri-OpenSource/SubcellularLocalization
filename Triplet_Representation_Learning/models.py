import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as initialization_strategy
from torch_geometric.data import Data
import pdb

activation_dictionary = {'Linear' : None, 'RELU' : F.relu, 'Tanh' : F.tanh, 'Sigmoid' : F.sigmoid, 'GELU' : F.gelu, 'SILU' : F.silu, 'ELU' : F.elu}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=42
def seed_torch():
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    return None

class TripletEmbeddingDataset(torch.utils.data.Dataset):

    def __init__(self, triplet_list, embedding_df, device):

        super(TripletEmbeddingDataset, self).__init__()

        self.triplet_list = triplet_list
        self.embedding_df = embedding_df
        self.device = device

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):

        triplet = self.triplet_list[idx]

        anchor = torch.tensor(self.embedding_df.loc[triplet[0]]).to(self.device)
        positive = torch.tensor(self.embedding_df.loc[triplet[1]]).to(self.device)
        negative = torch.tensor(self.embedding_df.loc[triplet[2]]).to(self.device)

        return (anchor, positive, negative)

class TripletNet(nn.Module):
    def __init__(self, hyperparameter_dict):

        super(TripletNet, self).__init__()

        seed_torch()

        self.device = device
        self.alpha = 1.0

        self.input_dim = 1024
        self.n_linear_layers = hyperparameter_dict['n_linear_layers']
        self.decay_rate = hyperparameter_dict['decay_rate']
        self.embedding_dim = hyperparameter_dict['embedding_dim']

        in_features_list = []
        for layer_num in range(self.n_linear_layers):

            if layer_num == 0:
                in_features_list.append(self.input_dim)

            elif layer_num == (self.n_linear_layers - 1):
                in_features_list.append(self.embedding_dim)

            else:
                n_layer_features = int(np.floor((in_features_list[-1] * self.decay_rate)))
                
                if n_layer_features < self.embedding_dim:
                    n_layer_features = self.embedding_dim
                
                in_features_list.append(n_layer_features)

        linear_layers = nn.ModuleList()
        batch_norms = nn.ModuleList()

        for layer_num in range(self.n_linear_layers):
                
                if layer_num == 0:
                    linear = nn.Linear(self.input_dim, in_features_list[layer_num])

                else:
                    linear = nn.Linear(in_features_list[layer_num - 1], in_features_list[layer_num])

                if linear.weight is not None:
                    initialization_strategy.xavier_uniform_(linear.weight)
                if linear.bias is not None:
                    linear.bias.data.fill_(0.01)

                batch_norm = nn.BatchNorm1d(in_features_list[layer_num]) 

                linear_layers.append(linear)
                batch_norms.append(batch_norm)

        self.linear_layers = linear_layers
        self.batch_norms = batch_norms

        self.activation_function = activation_dictionary[hyperparameter_dict['activation_function']]
        self.dropout_layer = nn.Dropout(p=hyperparameter_dict['dropout_weight'])    

        seed_torch()

    def embed(self, tensor):

        seed_torch()

        for layer_num in range(len(self.linear_layers)):

            if layer_num == 0:
                embeddings = self.linear_layers[layer_num](tensor)
            
            else:
                embeddings = self.linear_layers[layer_num](embeddings)
            
            embeddings = self.batch_norms[layer_num](embeddings)

            if self.activation_function != None:
                embeddings = self.activation_function(embeddings)

            if layer_num != (len(self.linear_layers) - 1):
                embeddings = self.dropout_layer(embeddings)

        return embeddings

    def forward(self, triplet_data):

        seed_torch()
        
        return self.embed(triplet_data[0]), self.embed(triplet_data[1]), self.embed(triplet_data[2])

    def compute_triplet_loss(self, anchor, positive, negative):

        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        loss = torch.relu(distance_positive - distance_negative + self.alpha)
    
        return loss.mean()