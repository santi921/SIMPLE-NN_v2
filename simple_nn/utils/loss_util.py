import warnings
import numpy as np 
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn.modules.loss import _Loss
import torch.nn._reduction as _Reduction
import torch.nn.functional as F
from torch.autograd import Variable


class cutoffMSE(_Loss):
    #__constants__ = ['reduction']

    def __init__(self, size_average=0, weight: Tensor = Tensor([0])) -> None:
        self.weight = weight
        super(cutoffMSE, self).__init__(size_average)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor = Tensor([0])) -> Tensor:
        return mse_cutoff_loss(input, target, weight = self.weight) # reduction=self.reduction

def mse_cutoff_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor = Tensor([0])
) -> Tensor:

    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()),
            stacklevel=2,
        )
    
    
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    size = expanded_target.size()
    
    mse_plain = (input - target)**2
    
    weights = torch.zeros_like(expanded_target)
    mse_sorted, index = torch.sort(mse_plain)

    try:
        weights[0,0] = 1
        weights[0,1] = 1
        weights[0,2] = 1
        weights[0,3] = 1
    except:
        try:
            weights[0] = 1
            weights[1] = 1
            weights[2] = 1
            weights[3] = 1
        except:
            try:
                weights[0,0,0] = 1
                weights[0,0,1] = 1
                weights[0,0,2] = 1
                weights[0,0,3] = 1
            except:
                print(weights)
    out = mse_plain * weights
    loss_out = nn.MSELoss(reduction='none')
    zeros = torch.zeros_like(out)
    return loss_out(out, zeros)
