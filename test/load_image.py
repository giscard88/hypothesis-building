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

def main():
    threshold_={0:0.3,1:0.7,2:0.57,3:0.57, 4:0.88}

    image_num=500 # the number of splited intermediate images
    #layer_num=1  # the layer # of monitoring  [conv, layer1, layer2, layer3, linear]
    for ln in [0]: #range(layer_num):
        flag_init=True
        for im in range(image_num):
            print (ln,im)
            inputs=torch.load('image/hook'+str(im)+'_'+str(ln)+'.th',map_location=lambda storage, loc: storage) #inputs (batch_num,input_size)
            size=inputs.size()
            if flag_init:
                cog=CogMem_torch(size[1],threshold_[ln])
                flag_init=False
                
            for ii in range(size[0]):
                #print (h,ii)
                cog.Test_batch(inputs[ii,:])
        
        print (cog.wm.size())    
        torch.save(cog.wm,'wm_'+str(ln)+'.pt')                







if __name__ == '__main__':
    main()
