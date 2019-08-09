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

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=50000, shuffle=False,**kwargs)   
    for data, target in train_loader:
        label=target.numpy()
      

    
    layer_num=5
    image_num=500
    pred_n=torch.load('prediction_resnet.pt',map_location=lambda storage, loc: storage)
    print (pred_n)
    
    for ln in [0]:
        wm=torch.load('wm_'+str(ln)+'.pt',map_location=lambda storage, loc: storage)
        
        cog=CogMem_load(wm.to(device),label) 
        
        for im in range(image_num):
            print (ln,im)
            
            temp=torch.load('image/hook'+str(im)+'_'+str(ln)+'.th',map_location=lambda storage, loc: storage)
            if im==0:
                inputs=temp
                
            else:
                inputs=torch.cat((inputs, temp), 0)     
                
            del temp
        
        cog.forward(inputs.to(device))
        temp=make_association(cog.image,pred_n,device)
        
        
        torch.save(temp,'map_association_'+str(ln)+'.pt')

        del temp, inputs, wm, cog
    
    
   
          
    
        
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

