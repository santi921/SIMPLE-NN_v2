import sys
sys.path.insert(0, '../../../')
from simple_nn import simple_nn
from simple_nn.init_inputs import initialize_inputs
from simple_nn.models import run

import torch
from torch.nn import Linear

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']

torch.set_default_dtype(torch.float64)
device = run._set_device()
model = run._initialize_model(inputs, logfile, device)
optimizer = run._initialize_optimizer(inputs, model)
checkpoint, loss = run._load_checkpoint(inputs, logfile, model, optimizer)

scale_factor, pca = run._load_scale_factor_and_pca(inputs, logfile, checkpoint)
print(scale_factor)
print(pca)
