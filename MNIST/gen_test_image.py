import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from noname import *
import numpy as np
import json
import os
from Nets import *



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  
 
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()





    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dr_t='./data'
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=False, **kwargs)

    model=Net()
    checkpoint = torch.load('pretrained_models/mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    
    model.to(device)

    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]
    
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    #hookF=[Hook(layer [1]) for layer in list(model._modules.items())]
    #test=[layer[1] for layer in list(model._modules.items())]
    #for layer in list(model._modules.items()):
    #for xin in param_dict:
    #    print (xin)
    
    scan_test_export(args, model, device, test_loader, hookF)
   
    #for hook in hookF:
    #    print (hook.output.size())

    #for xin in model._modules.items():
    #    print ((xin[1]).in_channels)
       
if __name__ == '__main__':
    main()
    del internal




