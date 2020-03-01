import argparse
import os
import time


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
from Nets import *



split_factor=1

def load_test_image(sel_layer):
    flag=True
    for ba in range(split_factor):
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

  
    parser.add_argument('--layer', type=int, default=0, metavar='N',
                        help='select a layer 0-3, which represents 2 CNs and 2 FCs')

#    parser.add_argument('--threshold', type=int, default=0, metavar='N',
#                        help='threshold values used when cogmemry is generated')
    

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 



    dr_t='./data'
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=False, **kwargs)


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

   
    
    
   
    pred_n=torch.load('test_prediction_CNN.pt',map_location=lambda storage, loc: storage) 
    
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
    
    if os.path.exists("confusion_all_2"): # comprehensive confusion
        pass
    else:
        os.mkdir("confusion_all_2")
   
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
        cog=CogMem_load(wm) 
        #cog=CogMem_load(wm,label)


        
        cog.forward(roV)
        #pred=cog.pred.long()
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
        cnt=0
        #print ('sel shape',sel.shape)
        #print (cog.image.size())       
        for xi, xin in enumerate(pred_n):
            cls=xin.item()
            label_t=labels[xi].long().item()
            v2=cog.image[:,xi]
            #mem.append(v2.cpu().numpy())
            idx=torch.argsort(v2).cpu().numpy()
            idx=np.flip(idx,0)[:3]
   
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
            if idx2==label_t:
                c1=idx2            
                cnt=cnt+1
                for c2 in range(10):
                    #if c1!=c2:
                    corr[c1,c2]=corr[c1,c2]+np.exp(temp_v[c2])/np.exp(temp_v[c1])
        print (cnt)
        for c1 in range(10):
            corr[c1,:]=corr[c1,:]/float(cnt)
        for c1 in range(10):
            corr[c1,c1]=0
        max_v=np.amax(corr)
        #corr=corr/500.0         
        print (layer_sel_, total_1,total_2,total_3)
        print ('cons1',cons1,'cons2',cons2)
        #pylab.figure(ttin+1)
        pylab.imshow(corr,cmap='jet', vmax=max_v)
        pylab.colorbar() 
        pylab.savefig('confusion_all_2/L'+str(layer_sel_)+'_'+str(th)+'.png')
        pylab.savefig('confusion_all_2/L'+str(layer_sel_)+'_'+str(th)+'.eps')
        pylab.close()    
            
        temp_dict={'max_pred':total_1, 'ref_accuracy':total_2, 'max_accuracy':total_3,'consist_pred':cons1, 'consist_accuracy':cons2, 'cog_size':wm.size()}
        #mem=np.array(mem)
        #np.savetxt('mem_'+str(layer_sel_)+'.txt',mem)
        torch.cuda.empty_cache() 
        del cog, roV, sel, wm, act_map
        results[str(th)]=temp_dict
    fp=open('confusion_all_2/prediction'+str(layer_sel_)+'.json','w')
    json.dump(results,fp)
    fp.close  
    #pylab.show()

    
    
    
    
   
        
      
        
        

if __name__ == '__main__':
    main()
    

