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
from resnet import *


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 60000)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if os.path.exists('/local2'):
        dr_t='local2/data'
    else:
        dr_t='/home/jung/hypothesis/data'
    train_loader=torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
            
        ]), download=True),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.test_batch_size, shuffle=False,**kwargs)

    model=resnet44()
    model.to(device)

    checkpoint = torch.load('pretrained_models/resnet44.th',map_location=lambda storage, loc: storage)
    states=checkpoint['state_dict']
    del checkpoint
    param_dict={}


    for pr in states:
        param_dict[pr[7:]]=states[pr]

    model.load_state_dict(param_dict)
    hookF=[Hook(model.conv1), Hook(model.layer1),Hook(model.layer2),Hook(model.layer3),Hook(model.linear)]
    
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




