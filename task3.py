import torch
import torch.nn as nn

class Rumelhart1986Net(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.hidden_layer.weight.uniform_(-0.3, 0.3)
            self.hidden_layer.bias.uniform_(-0.3, 0.3)
            self.output_layer.weight.uniform_(-0.3, 0.3)
            self.output_layer.bias.uniform_(-0.3, 0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.sigmoid(self.hidden_layer(x))
        out = self.sigmoid(self.output_layer(h))
        return out

def compute_rumelhart_loss(predictions, targets, use_tolerance=True):
    temp_loss = 0.0 
    if use_tolerance:
        mask = torch.ones_like(predictions)
        mask[(targets == 1.0) & (predictions > 0.8)] = 0.0
        mask[(targets == 0.0) & (predictions < 0.2)] = 0.0
        diff = mask * (predictions - targets)
        squared_diff = diff ** 2
    else:
        squared_diff = (predictions - targets) ** 2
        
    return 0.5 * torch.sum(squared_diff)

class RumelhartMomentumOptimizer:
    def __init__(self, models_params, lr=0.1, alpha=0.9, weight_decay=0.002):
        self.params = list(models_params)
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.velocities = []
        for p in self.params:
            self.velocities.append(torch.zeros_like(p.data))

    def step(self):
        with torch.no_grad():
            for i in range(len(self.params)):
                p = self.params[i]
                v = self.velocities[i]
                
                if p.grad is None:
                    continue
                
                grad = p.grad.data.clone()
                
                if type(self.lr) == int:
                    self.lr = float(self.lr)
                    
                v.copy_((-self.lr * grad) + (self.alpha * v))
                p.data.add_(v)
                
                if self.weight_decay > 0.0:
                    p.data.mul_(1.0 - self.weight_decay)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()