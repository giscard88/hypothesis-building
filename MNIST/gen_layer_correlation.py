import argparse
import os
import time




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

from advertorch.attacks import LinfPGDAttack

threshold_0=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] 
threshold_1=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] 
threshold_2=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
threshold_3=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
thresholds_=[threshold_0,threshold_1,threshold_2,threshold_3]


    
thres_set1=[3, 4, 5, 6]
thres_set2=[4, 5, 6, 7]
thres_set3=[3, 3, 3, 3]
thres_set4=[3, 5, 6, 7]
thres_set5=[4, 4, 5, 5]
thres_set6=[4, 4, 6, 6]
thres_set7=[4, 4, 7, 7]

set_list=[thres_set1, thres_set2, thres_set3, thres_set4, thres_set5, thres_set6, thres_set7]

def main():
    data=[]
    torch.cuda.empty_cache() 

    parser = argparse.ArgumentParser(description='correlations among intermediate layers')
   
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--set', type=int, default=1,
                        help='select the set of thresholds')

    parser.add_argument('--eps', default=0.3,
                        help='eps for LinfPGDAttack')

    parser.add_argument('--norm', action='store_true', default=False,
                        help='adversarial?')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
 

    dr_t='./data'

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(dr_t, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=200, shuffle=False, **kwargs)

    set_=set_list[args.set]

    model=Net()
    checkpoint = torch.load('pretrained_models/mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    
    model.to(device)
    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]


    for data, target in test_loader:
        break
    labels_=target
    labels_numpy=[]
    for xi in labels_:
        temp_=np.zeros(10)
        temp_[xi.item()]=1.0
        labels_numpy.append(temp_)
    labels_numpy=np.array(labels_numpy)


    data=data.to(device)
    labels_=labels_.to(device)

    if args.norm:
        inputs=torch.load('adversarial_examples/norm.pt',map_location=lambda storage, loc: storage)
        fn='norm'
        
    else:
        inputs=torch.load('adversarial_examples/adv_'+str(args.eps)+'.pt',map_location=lambda storage, loc: storage)
        fn='adv'
    inputs=inputs.to(device)
    pred_n=test_adv(args, model, device, test_loader, hookF, inputs)

    
    Associations=[]
    for xin in range(4):
        thres=thresholds_[xin][set_[xin]]
        temp=torch.load('coglayer/map_association_'+str(xin)+'_'+str(thres)+'.pt',map_location=lambda storage, loc: storage)
        Associations.append(temp)



    activity_layer={}
    for ttin in range(4):
        layer_sel_=ttin
        
        thres=thresholds_[ttin][set_[ttin]]
        roV=intermediate_output[layer_sel_]

        sel=Associations[layer_sel_].map
        sel=sel.cpu().numpy()

        wm=torch.load('coglayer/wm_'+str(layer_sel_)+'_'+str(thres)+'.pt',map_location=lambda storage, loc: storage)
        

        cog=CogMem_load(wm,None) 



        for data, target in test_loader:
            break
        labels_=target
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
        mem=[]
        #print ('sel shape',sel.shape)
        #print (cog.image.size())
        #print ('pred',pred.size())
        temp_vec=[]       
        for xi, xin in enumerate(pred_n):
            cls=xin.item()
            label_t=labels_[xi].long().item()
            v2=cog.image[:,xi]
       
            idx=torch.argsort(v2).cpu().numpy()
            #mem.append(v2.cpu().numpy())
            idx=np.flip(idx,0)[:20]
            tar=sel[idx,:]
            temp_v=np.zeros(10)
            for zin in idx:
                temp_v=temp_v+sel[zin,:]*v2[zin].item()
            temp_v=np.exp(temp_v)
            temp_v=temp_v/(np.sum(temp_v)+np.finfo(float).eps)
            temp_vec.append(temp_v)
            
        activity_layer[layer_sel_]=np.array(temp_vec)
  
    layer_corr={}
    values=[]    
    for xin in activity_layer:
        pred=activity_layer[xin]
        for yin in activity_layer:
            post=activity_layer[yin]
            if xin==yin:
                pass
            else:
                temp=[]
                for zin in range(len(pred_n)):
                    temp.append(np.dot(pred[zin],post[zin])/(np.linalg.norm(pred[zin])*np.linalg.norm(post[zin])+np.finfo(float).eps))
                #temp=np.array(temp)
                layer_corr[str(xin)+'_'+str(yin)]=temp


    if os.path.exists("correlations"):
        pass
    else:
        os.mkdir("correlations")                
    if fn=='adv':
        fp=open('correlations/layer-'+fn+'_'+str(args.set)+'-'+str(args.eps)+'.json','w')
        json.dump(layer_corr,fp)
    else:
        fp=open('correlations/layer-'+fn+'_'+str(args.set)+'.json','w')
        json.dump(layer_corr,fp)


    
                    
                       
    
   
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

