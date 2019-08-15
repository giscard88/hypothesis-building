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
#from Nets import *
from resnet import *

parser = argparse.ArgumentParser(description='construct the cognitive memory')

parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='select a layer 0-4, which represents the first CNN, 3 composite layers and the final FC')

args = parser.parse_args()

def main():

    if os.path.exists("coglayer"):
        pass
    else:
        os.mkdir("coglayer")
  

    #threshold_={0:0.3,1:0.75,2:0.6,3:0.57, 4:0.88}
    threshold_0=[0.36] #[0.2, 0.22, 0.24, 0.26, 0.3, 0.32, 0.34]
    threshold_1=[0.70, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84]
    threshold_2=[0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66]
    threshold_3=[0.5,  0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64]
    threshold_4=[0.8,  0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94]
    image_num=500 # the number of splited intermediate images
    #layer_num=1  # the layer # of monitoring  [conv, layer1, layer2, layer3, linear]
    ln=args.layer
    if ln==0:
        thresholds_=threshold_0
    elif ln==1:
        thresholds_=threshold_1
    elif ln==2:
        thresholds_=threshold_2
    elif ln==3:
        thresholds_=threshold_3
    elif ln==4:
        thresholds_=threshold_4

    for th in thresholds_:
        flag_init=True
        for im in range(image_num):
            print (ln,im)
            inputs=torch.load('image/hook'+str(im)+'_'+str(ln)+'.th',map_location=lambda storage, loc: storage) #inputs (batch_num,input_size)
            size=inputs.size()
            
            if flag_init:
                cog=CogMem_torch(size[1],th)
                flag_init=False
                
            for ii in range(size[0]):
                #print (h,ii)
                cog.Test_batch(inputs[ii,:])
        
        print (cog.wm.size())    
        torch.save(cog.wm,'coglayer/wm_'+str(ln)+'_'+str(th)+'.pt')                







if __name__ == '__main__':
    main()
