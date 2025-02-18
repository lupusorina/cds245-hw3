import torch
import torch.nn as nn

class BCModel(nn.Module):
    def __init__(self, input_size=3, output_size=1):
        super(BCModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, output_size),
        )

    def forward(self, x):
        output = self.net(x)
        return 2 * torch.tanh(output) # Pendulum-v1 action_space is -2 to 2