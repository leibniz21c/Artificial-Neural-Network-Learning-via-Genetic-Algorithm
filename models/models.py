import torch.nn as nn


class NNModel(nn.Module):
    def __init__(self, weight_scaling=0.1):
        super(NNModel, self).__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=4, out_features=256, bias=False),
            nn.Linear(in_features=256, out_features=2, bias=False),
        )
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

        # Weight scaling
        for layer in self.linear_layers:
            layer.weight.data *= weight_scaling

        # Model
        self.eval()

    def forward(self, x):
        for layer in self.linear_layers:
            x = self.relu(layer(x))
        return self.softmax(x)

    @property
    def weight(self):
        return [layer.weight.data for layer in self.linear_layers]

    @weight.setter
    def weight(self, w):
        for i, layer in enumerate(self.linear_layers):
            layer.weight.data = w[i]
