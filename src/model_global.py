import torch
from torchvision import models
import torch.nn as nn
from collections import OrderedDict

def global_model(state_dict_path):
    model = models.densenet201(pretrained=False)
    model.classifier = nn.Sequential(OrderedDict([
        ('fcl1', nn.Linear(1920,6)),
        ('out', nn.Sigmoid()),
    ]))

    state_dict = torch.load(state_dict_path, map_location = 'cpu')['state_dict']
    for keyA, keyB in zip(state_dict, model.state_dict()):
        state_dict = OrderedDict((keyB if k == keyA else k, v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)
    
    return model
    