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
from Nets import *


def main():
    data=[]
    labels=[]

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
    kwargs = {'num_workers': 1 } if use_cuda else {}

    if os.path.exists('/local2'):
        dr_t='local2/data'
    else:
        dr_t='/home/jung/hypothesis/data'

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=False, **kwargs)
    model=Net()
    model.to(device)
    checkpoint = torch.load('mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)


    
    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]
    test(args, model, device, test_loader, hookF)
    data_wm=[]
    for xin in [0,1,2,3]:
        wm=torch.load('wm_'+str(xin)+'.pt',map_location=lambda storage, loc: storage)
        data_wm.append(wm)
        fp=open('labels_'+str(xin)+'.json')
        label=json.load(fp)
        fp.close()
        cog=CogMem_load(wm,label)
        roV=intermediate_output[xin].cpu() # cog expects CPU vectors.
        cog.foward(roV)
        pred=cog.pred.long()
        print (pred)
        for data, target in test_loader:
            print (target)
            a=pred.eq(target.view_as(pred)).sum().item()    
        
        print (xin, a)

if __name__ == '__main__':
    main()
    del intermediate_output

