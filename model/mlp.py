import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    """
    Represents standard MLP block.
    The article embeddings get passed into this block for number of reads prediction.
    """
    def __init__(self):
        super(MLPBlock, self).__init__()

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 1) # Output layer

    def forward(self, x):
        """ Perform a standard forward pass. """
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)        
        
        return x
