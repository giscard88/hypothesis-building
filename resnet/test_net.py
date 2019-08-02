from __future__ import print_function
import argparse
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
    parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                        help='input batch size for training (default: 60000)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
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

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if os.path.exists('/local2'):
        dr_t='local2/data'
    else:
        dr_t='/home/jung/hypothesis/data'
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dr_t, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(dr_t, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model=resnet(depth=110, num_classes=10)
    model.to(device)
    #for layer in list(model._modules.items()):
    #    print (layer)
    hookF=[Hook(model.layer1), Hook(model.layer2), Hook(model.layer3)]

    
    checkpoint = torch.load('cifar10_nest.pth.tar',map_location=lambda storage, loc: storage)
    
    states=checkpoint['state_dict']
    
    param_dict={}

    for pr in states:
        param_dict[pr[7:]]=states[pr]



    model.load_state_dict(param_dict)
    
    '''
    model.to(device)
    
   
    scan_train(args, model, device, train_loader, hookF)
    print (internal)
    for i in internal:
        print (i, internal[i].wm.size(),len(internal[i].labels_))
        torch.save(internal[i].wm,'wm_'+str(i)+'.pt')
        fp=open('labels_'+str(i)+'.json','w')
        json.dump(internal[i].labels_,fp)
        fp.close()
    #for hook in hookF:
    #    print (hook.output.size())

    #for xin in model._modules.items():
    #    print ((xin[1]).in_channels)
    '''
        
if __name__ == '__main__':
    main()
    del internal

