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

    parser = argparse.ArgumentParser(description='generation association between cog mem and prediction of CNN trained with MNIST')
  
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
  
    parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='select a layer 0-3, which represents 2 CNs and 2 FCs')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

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

   
    for data, target in train_loader:
        label=target.numpy()
    
    coglayer=args.layer
    #threshold_={0:0.3,1:0.75,2:0.6,3:0.57, 4:0.88}
    threshold_0=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] 
    threshold_1=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] 
    threshold_2=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    threshold_3=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    if coglayer==0:
        thresholds_=threshold_0
    elif coglayer==1:
        thresholds_=threshold_1
    elif coglayer==2:
        thresholds_=threshold_2
    elif coglayer==3:
        thresholds_=threshold_3

    
    layer_num=4
    image_num=1
    pred_n=torch.load('prediction_CNN.pt',map_location=lambda storage, loc: storage)
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

