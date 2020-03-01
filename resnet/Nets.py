from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from noname import *
import numpy as np
import json

internal={}
intermediate_output={}
labels_=[]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    


def scan_train(args, model, device, loader,hookF):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for h, hook in enumerate(hookF):
                a_int=hook.output
                a_size=a_int.size()
                a_int=a_int.view(a_size[0],-1)
                a_size=a_int.size()
                if h not in internal:
                    threshold=0.65+0.05*float(h)
                    internal[h]=CogMem_torch(a_size[1],threshold)
                for ii in range(a_size[0]):
                    print (h,ii)
                    internal[h].Test_batch(a_int[ii,:], target[ii].item())
                
                 
                

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

def scan_train_export(args, model, device, loader,hookF):
    model.eval()
    test_loss = 0
    ct = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            for h, hook in enumerate(hookF):
                a_int=hook.output
                a_size=a_int.size()
                a_int=a_int.view(a_size[0],-1)
                torch.save(a_int,'image/hook'+str(ct)+'_'+str(h)+'.th')
            ct=ct+1

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

def scan_test_export(args, model, device, loader,hookF):
    model.eval()
    test_loss = 0
    ct = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            for h, hook in enumerate(hookF):
                a_int=hook.output
                a_size=a_int.size()
                a_int=a_int.view(a_size[0],-1)
                torch.save(a_int,'test_image/hook'+str(ct)+'_'+str(h)+'.th')
            ct=ct+1

    test_loss /= len(loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

                

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
       
    def close(self):
        self.hook.remove()

def test(args, model, device, test_loader,hookF):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for h, hook in enumerate(hookF):
                a_int=hook.output
                a_size=a_int.size()
                a_int=a_int.view(a_size[0],-1)
                a_size=a_int.size()
                 
                
                intermediate_output[h]=a_int
                
                 
                

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return pred

def prediction(args, model, device, test_loader,hookF):
    model.eval()
    test_loss = 0
    correct = 0
    
    ct=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
           
            print (ct)
            if ct==0:
                preds_=pred
            else:
                preds_=torch.cat((preds_,pred),0)
            ct=ct+1
            
    return preds_

def test_adv(args, model, device, test_loader,hookF,adv_input):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            break
        data, target = data.to(device), target.to(device)
        data=adv_input
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        for h, hook in enumerate(hookF):
            a_int=hook.output
            a_size=a_int.size()
            a_int=a_int.view(a_size[0],-1)
            a_size=a_int.size()
                 
                
            intermediate_output[h]=a_int.cpu()
                
                 
                

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return pred
    


class make_association:
    def __init__(self, pre, post,device,class_n=10):
        self.pre=pre
        self.post=post
        self.pre_n, self.inst_n=pre.size()
        self.pos_n=class_n
        self.device=device
        #self.map=np.zeros((self.pos_n,self.pre_n))
        self.Convert()
        self.Evolve()

    def Convert(self):
        self.target_m=np.ones((self.inst_n,self.pos_n))*-1.0

        for xin in range(self.inst_n):

            self.target_m[xin,int(self.post[xin].item())]=1.0
           
        self.target_m=torch.from_numpy(self.target_m)
       
        
    def Evolve(self):
        
        self.target_m=self.target_m.float() # it seems that numpy.ones has dtype=double.
        self.pre=self.pre-1
        self.pre=self.pre/0.01
        self.pre=self.pre.exp()
        
        delta=torch.matmul(self.pre.to(self.device),self.target_m.to(self.device))
        self.map=delta


