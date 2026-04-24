import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.gate_scores, mean=2.0, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()


class SelfPruningMLP(nn.Module):
    
    def __init__(self, hidden1: int = 256, hidden2: int = 128):  # was 128, 64
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, hidden1)
        self.fc2 = PrunableLinear(hidden1, hidden2)
        self.fc3 = PrunableLinear(hidden2, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3]

    def all_gates(self) -> torch.Tensor:
        return torch.cat(
            [layer.gate_values().flatten() for layer in self.prunable_layers()]
        )

    def sparsity_loss(self) -> torch.Tensor:
        # L1 on all gates
        return sum(layer.gate_values().sum() for layer in self.prunable_layers())