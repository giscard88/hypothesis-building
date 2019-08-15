import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import pylab

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





def load_test_image(sel_layer):
    flag=True
    for ba in range(100):
        temp=torch.load('test_image/hook'+str(ba)+'_'+str(sel_layer)+'.th',map_location=lambda storage, loc: storage)
        if flag:
            im=temp
            flag=False
        else:
            im=torch.cat((im,temp),0)
    return im
 


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='prediction from cognitive memory')
  
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='select a layer 0-4, which represents the first CNN, 3 composite layers and the final FC')

#    parser.add_argument('--threshold', type=int, default=0, metavar='N',
#                        help='threshold values used when cogmemry is generated')
    

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=10000, shuffle=False,**kwargs)


    coglayer=args.layer
    #threshold_={0:0.3,1:0.75,2:0.6,3:0.57, 4:0.88}
    threshold_0=[0.2,  0.22, 0.24, 0.26, 0.3, 0.32, 0.34, 0.36]
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
    
   
    
    
   
    pred_n=torch.load('test_prediction_resnet.pt',map_location=lambda storage, loc: storage) 
    
    print (pred_n.size())
    
    for data, target in test_loader:   
        labels=target
    labels_=[]
    for xi in labels:
        temp_=np.zeros(10)
        temp_[xi.item()]=1.0
        labels_.append(temp_)
    labels_=np.array(labels_)
    print ('label.shape',labels_.shape) 
    
    if os.path.exists("confusion"):
        pass
    else:
        os.mkdir("confusion")
   
    results={}    
    for th in thresholds_:
        layer_sel_=coglayer
        temp_dict={}

        roV=load_test_image(layer_sel_)
        roV=roV.to(device)
        act_map=torch.load('coglayer/map_association_'+str(coglayer)+'_'+str(th)+'.pt',map_location=lambda storage, loc: storage)
        sel=act_map.map
        sel=sel.cpu().numpy()

        wm=torch.load('coglayer/wm_'+str(layer_sel_)+'_'+str(th)+'.pt',map_location=lambda storage, loc: storage)
        
        wm=wm.to(device)      
        cog=CogMem_load(wm,labels_) 
        #cog=CogMem_load(wm,label)


        
        cog.forward(roV)
        pred=cog.pred.long()
        #pred=cog.pred.long().cpu().numpy()


    

        total_1=0
        total_2=0
        total_3=0
        total_4=0
        total_5=0 
        cons1=0
        cons2=0
        temp=0
        corr=np.zeros((10,10))
        #mem=[]
        #print ('sel shape',sel.shape)
        #print (cog.image.size())       
        for xi, xin in enumerate(pred_n):
            cls=xin.item()
            label_t=labels[xi].long().item()
            v2=cog.image[:,xi]
            #mem.append(v2.cpu().numpy())
            idx=torch.argsort(v2).cpu().numpy()
            idx=np.flip(idx,0)[:8]
   
            temp_v=np.zeros(10)
            for zin in idx:
                temp_v=temp_v+sel[zin,:]*v2[zin].item()
        

            idx2=np.argmax(temp_v)
            idx3=np.argsort(temp_v)
            idx3=np.flip(idx3,0)[:3]
            sum_v=np.sum(np.exp(temp_v))
        
            #print (xi, idx, cls, idx3, idx2)
            # cls: prediction, idx2: max from association, idx3, label from truth, idx_truth: ground truth
            if cls==idx2:
                total_1=total_1+1
            if label_t==cls:
                total_2=total_2+1

            if label_t==idx2:
                total_3=total_3+1

            if label_t!=cls:
                temp=temp+1
                if cls==idx2:
                    total_4=total_4+1
            else:
                temp=temp+1
                if cls==idx2:
                    total_5=total_5+1
       
            if cls in idx3:
                cons1=cons1+1
            if label_t in idx3:
                cons2=cons2+1
            for c1 in idx3:
                if c1==cls:
                    for c2 in idx3:
                        if c1!=c2:
                            corr[c1,c2]=corr[c1,c2]+np.exp(temp_v[c2])/sum_v

        max_v=np.amax(corr)
        #corr=corr/500.0         
        print (layer_sel_, total_1,total_2,total_3)
        print ('cons1',cons1,'cons2',cons2)
        #pylab.figure(ttin+1)
        pylab.imshow(corr,cmap='jet', vmax=125.0)
        pylab.colorbar() 
        pylab.savefig('confusion/L'+str(layer_sel_)+'_'+str(th)+'.png')    
            
        temp_dict={'max_pred':total_1, 'ref_accuracy':total_2, 'max_accuracy':total_3,'consist_pred':cons1, 'consist_accuracy':cons2, 'cog_size':wm.size()}
        #mem=np.array(mem)
        #np.savetxt('mem_'+str(layer_sel_)+'.txt',mem)
        torch.cuda.empty_cache() 
        del cog, roV, sel, wm, act_map
        results[str(th)]=temp_dict
    fp=open('confusion/prediction'+str(layer_sel_)+'.png','w')
    json.dump(results,fp)
    fp.close  
    #pylab.show()

    
    
    
    
   
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

