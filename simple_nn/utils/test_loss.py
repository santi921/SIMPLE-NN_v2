from loss_util import * 
from torch import nn
import torch
input = torch.randn(3, 5)
target = torch.randn(3, 5, requires_grad = True)


loss = nn.MSELoss(reduction='none')
output = loss(input, target)


loss2 = cutoffMSE()
output2 = loss2(input, target)
output2

print(output)
print(output2)
