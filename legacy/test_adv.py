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
import pylab
from numpy import linalg as LA
from collections import defaultdict

from advertorch.attacks import LinfPGDAttack



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
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader,hookF,adv_input):
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
    

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.output = output
       
    def close(self):
        self.hook.remove()

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
        self.target_m=np.ones((self.inst_n,self.pos_n))*0.0

        for xin in range(self.inst_n):

            self.target_m[xin,int(self.post[xin].item())]=1.0
           
        self.target_m=torch.from_numpy(self.target_m).to(self.device)
       
        
    def Evolve(self):
        
        self.target_m=self.target_m.float() # it seems that numpy.ones has dtype=double.
        self.pre=self.pre-1
        self.pre=self.pre/0.1
        self.pre=self.pre.exp()
        
        delta=torch.matmul(self.pre,self.target_m)
        self.map=delta
    
    '''
    def Evolve(self):
        
        self.target_m=self.target_m.float() # it seems that numpy.ones has dtype=double.
        temp=torch.zeros(self.pre_n,self.inst_n)
        idx=torch.argmax(self.pre,dim=0)
        #print (idx)
        for xin in range(self.inst_n):
            temp[idx[xin].item(),xin]=self.pre[idx[xin].item(),xin]
        
        
        delta=torch.matmul(temp,self.target_m)
        self.map=delta
    '''

        
           
              
        
        


def main():
    data=[]
    torch.cuda.empty_cache() 

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1 } if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=60000, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=10000, shuffle=False, **kwargs)

    test_loader_small = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=False, **kwargs)

    strage=device
    model=Net()
    model.to(device)
    checkpoint = torch.load('mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)

    layer_sel=3
    Associations=[]
    for xin in range(4):
        temp=torch.load('map_association_'+str(xin)+'.pt')
        Associations.append(temp)
    sel=Associations[layer_sel]
    sel=sel.numpy()
    #pylab.figure(1)
    



    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]

    
    # Let's test how this association is predictive of the test set

    
    data_wm=[]
    for xin in [layer_sel]:
        wm=torch.load('wm_'+str(xin)+'.pt',map_location=lambda storage, loc: storage)
        data_wm.append(wm)
        fp=open('labels_'+str(xin)+'.json') # labels for wm i.e., the labels of the test set.
        label=json.load(fp)
        fp.close()
        cog=CogMem_load(wm,label) 

    
    
    act_map=Associations[layer_sel]

    adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.8,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

    for data, target in test_loader_small:
        break
    labels_=target
    data=data.to(device)
    labels_=labels_.to(device)

    #adv_untargeted = adversary.perturb(data, labels_)
    adv_untargeted=data
    print (adv_untargeted.size())

    pred_n=test(args, model, device, test_loader_small, hookF, adv_untargeted)

    roV=intermediate_output[layer_sel]

    for data, target in test_loader_small:
        break
    labels_=target
    cog.foward(roV)
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
    mem=[]
    print ('sel shape',sel.shape)
    print (cog.image.size())
    print ('pred',pred.size())       
    for xi, xin in enumerate(pred_n):
        cls=xin.item()
        label_t=labels_[xi].long().item()
        v2=cog.image[:,xi]
       
        idx=torch.argsort(v2).cpu().numpy()
        mem.append(v2.cpu().numpy())
        idx=np.flip(idx,0)[:3]
        tar=sel[idx,:]
        temp_v=np.zeros(10)
        for zin in idx:
            temp_v=temp_v+sel[zin,:]*v2[zin].item()
        
        #print (temp_v)
        #tar=sel[idx,:]
        #idx3=cog.labels[idx].long().item()
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
    print ('pred. of prediction:', total_1,'global pred. of actual class:',total_2,'local pred. of actual class:', total_3)
    print ('cons1',cons1,'cons2',cons2)
    print (idx3)
           
            
        
    mem=np.array(mem)
    
    data=np.loadtxt('mem_'+str(layer_sel)+'.txt')
    temp=np.argsort(mem[0])
    temp=np.flip(temp,0)
    print ('adv',temp[:3])
    temp=np.argsort(data[0,:])
    temp=np.flip(temp,0)
    print ('clean',temp[:3])
    diff=data[0,:]-mem[0]
    #print ('diff',diff)
    print (np.amax(diff),np.mean(diff),np.std(diff),np.amin(diff))
    pylab.plot(data[0,:], label='clean')
    pylab.plot(mem[0], label='adv')
    pylab.legend() 
    pylab.show()
    
    torch.cuda.empty_cache() 
    del cog, roV


    
    
    
   
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

