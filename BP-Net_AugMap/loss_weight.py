import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, NamedTuple
    

class LOSS_WEIGHT(NamedTuple):
    loss: torch.nn.Module
    weight: float

    def to(self, device: torch.device):
        self.loss.to(device)
        return self

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.loss(*args, **kwds) * self.weight