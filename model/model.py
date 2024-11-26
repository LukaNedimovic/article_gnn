import torch
import torch.nn as nn
from torch.nn.functional import dropout

from model.gnn import GNNBlock
from model.mlp import MLPBlock


class BERTGNNModel(nn.Module):
    """
    Represents the complete BERT + GNN + MLP model.
    BERT embeddings are created apriori, 
    and are passed into the forward() method.
    """
    
    def __init__(self):
        super(BERTGNNModel, self).__init__()

        # Initialize layers
        self.gnn = GNNBlock()
        self.mlp = MLPBlock()
    
        # Port the model to adequate device (preferably CUDA, otherwise CPU)
        self.port_to_device()

    def port_to_device(self):
        """ Port the model to set device. """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        self.device = device
        
        print(f'Model ported to device: {device}')

    def forward(self, x, edge_index, id_2_idx, training: bool = True):
        """ Perform a standard forward pass. """
        
        # Infer embeddings
        x = self.gnn(x, edge_index, id_2_idx)
        x = dropout(
            x,
            p=0.3,
            training=training,
        )
        
        # Pass the embeddings through MLP to get the prediction for respective article
        x = self.mlp(x).view(-1)
        return x

    def __str__(self):
        return f'{self.gcn}\n{self.mlp}'