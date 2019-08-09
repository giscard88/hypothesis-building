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
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=50000, shuffle=False,**kwargs)   
    for data, target in train_loader:
        label=target.numpy()
    device = torch.device("cuda" if use_cuda else "cpu")    

    Assocations=[]
    layer_num=5
    image_num=500
    pred_n=np.loadtxt('prediction_resnet.txt')
    pred_n=torch.from_numpy(pred_n)
 
    for ln in range(layer_num):
        wm=torch.load('wm_'+str(xin)+'.pt',map_location=lambda storage, loc: storage)
        
        cog=CogMem_load(wm,label) 
        images=[]
        for im in range(image_num):
            inputs=torch.save('image/hook'+str(im)+'_'+str(ln)+'.th')     
            images.append(inputs.numpy())
        images=np.array(images)
        images=torch.from_numpy(images)
        cog.forward(images)
        temp=make_association(cog.image,pred_n,device)
        print (temp.map.size())
        Associations.append(temp.map)
        
    for xin in range(layer_num):
        data=Associations[xin]
        torch.save(data,'map_association_'+str(xin)+'.pt')
    '''

    # end of script    
          
    
        
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

