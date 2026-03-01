import torch as t
from torch import nn as nn

#some shi I copied off of chatGPT

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 16)
        self.l2 = nn.Linear(16, 16)
        self.l3 = nn.Linear(16, 16)
        self.l4 = nn.Linear(16, 10)

    def forward(self, x):
        x = t.relu(self.l1(x))
        x = t.relu(self.l2(x))
        x = t.relu(self.l3(x))
        x = t.relu(self.l4(x))
        return x


model = SimpleNN()
print(model)


