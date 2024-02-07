import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(240*320*3),
            nn.Linear(240*320*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        print()
        act = self.MLP(x)
        return act
    
    def get_loss(self, output, label):
        loss = ((output - label)**2).mean()
        return loss