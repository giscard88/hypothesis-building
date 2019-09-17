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


    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dr_t='./data'
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=60000, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=False, **kwargs)

    model=Net()
    checkpoint = torch.load('pretrained_models/mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    
    model.to(device)

    hookF=[Hook(model.conv1), Hook(model.layer1),Hook(model.layer2),Hook(model.layer3),Hook(model.linear)]
    
    pred_n=prediction(args, model, device, train_loader, hookF)
    
    #pred_n=np.array(pred_n)
    #pred_n=torch.from_numpy(pred_n)
    torch.save(pred_n,'prediction_CNN.pt') 

    pred_n=prediction(args, model, device, test_loader, hookF)
    
    #pred_n=np.array(pred_n)
    #pred_n=torch.from_numpy(pred_n)
    torch.save(pred_n,'test_prediction_CNN.pt') 
          
    
        
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

