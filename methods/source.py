from copy import deepcopy

import torch
import torch.nn as nn


class Source(nn.Module):
    """Source adapts a model by estimating feature statistics during testing."""

    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model.eval()

    def forward(self, x):
        return self.model(x)

    def reset(self):
        pass
