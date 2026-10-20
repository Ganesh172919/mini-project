"""Three supervised neural architectures used by the final submission."""
from torch import nn
from dl_common import MLP

class ResidualBlock(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(.2), nn.Linear(width, width))
    def forward(self, x):
        return (x + self.layers(x)).relu()

class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, 64)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.output = nn.Linear(64, 1)
    def forward(self, x):
        return self.output(self.blocks(self.input(x).relu())).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv1d(8, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(16*input_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1))).squeeze(-1)

def make_model(name, input_dim):
    return {'MLP': MLP, 'ResidualMLP': ResidualMLP, 'CNN1D': CNN1D}[name](input_dim)
