import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNBlock(nn.Module):
    """
    Represents standard GNN block, consisting of many GATConv layers,
    as written down about here:
    https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.nn.conv.GATConv.html
    
    Graph attentional operator from the "Graph Attention Networks" paper:
        @misc{veličković2018graphattentionnetworks,
        title={Graph Attention Networks}, 
        author={Petar Veličković and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Liò and Yoshua Bengio},
        year={2018},
        eprint={1710.10903},
        archivePrefix={arXiv},
        primaryClass={stat.ML},
        url={https://arxiv.org/abs/1710.10903}, 
        }
    """
    
    def __init__(self):
        super(GNNBlock, self).__init__()
        
        # Define 4 GATConv layers
        self.conv1 = GATConv(768, 1024, heads=1, concat=True) 
        self.conv2 = GATConv(1024, 2048, heads=1, concat=True)
        self.conv3 = GATConv(2048, 4096, heads=1, concat=True)
        self.conv4 = GATConv(4096, 8192, heads=1, concat=False)
        
    def forward(self, x, edge_index, id_2_idx):
        """ Perform a standard forward pass. """
        
        # Apply the node mapping
        for original, new_value in id_2_idx.items():
            edge_index[edge_index == original] = new_value

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
                
        return x