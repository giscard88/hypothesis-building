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

    parser = argparse.ArgumentParser(description='generation association between cog mem and prediction of resnet')
  
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='select a layer 0-4, which represents the first CNN, 3 composite layers and the final FC')
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
    
    coglayer=args.layer
    #threshold_={0:0.3,1:0.75,2:0.6,3:0.57, 4:0.88}
    threshold_0=[0.36] #[0.2,  0.22, 0.24, 0.26, 0.3, 0.32, 0.34, 0.36]
    threshold_1=[0.70, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84]
    threshold_2=[0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66]
    threshold_3=[0.5,  0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64]
    threshold_4=[0.8,  0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94]

    if coglayer==0:
        thresholds_=threshold_0
    elif coglayer==1:
        thresholds_=threshold_1
    elif coglayer==2:
        thresholds_=threshold_2
    elif coglayer==3:
        thresholds_=threshold_3
    elif coglayer==4:
        thresholds_=threshold_4  

    
    layer_num=5
    image_num=500
    pred_n=torch.load('prediction_resnet.pt',map_location=lambda storage, loc: storage)
    print (pred_n)
    
    for th in thresholds_:
        wm=torch.load('coglayer/wm_'+str(coglayer)+'_'+str(th)+'.pt',map_location=lambda storage, loc: storage)
        
        cog=CogMem_load(wm.to(device),label) 
        
        for im in range(image_num):
            print (th,im)
            
            temp=torch.load('image/hook'+str(im)+'_'+str(coglayer)+'.th',map_location=lambda storage, loc: storage)
            if im==0:
                inputs=temp
                
            else:
                inputs=torch.cat((inputs, temp), 0)     
                
            del temp
        
        cog.forward(inputs.to(device))
        temp=make_association(cog.image,pred_n,device)
        
        
        torch.save(temp,'coglayer/map_association_'+str(coglayer)+'_'+str(th)+'.pt')

        del temp, inputs, wm, cog
    
    
   
          
    
        
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

