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
        batch_size=200, shuffle=False, **kwargs)

    strage=device
    model=Net()
    model.to(device)
    checkpoint = torch.load('mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)


    adv_=True
    Associations=[]
    for xin in range(4):
        temp=torch.load('map_association_'+str(xin)+'.pt')
        Associations.append(temp)





    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]

    
    # Let's test how this association is predictive of the test set

    



    


    adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.7,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

    for data, target in test_loader_small:
        break
    labels_=target
    data=data.to(device)
    labels_=labels_.to(device)

    if adv_:
        adv_untargeted = adversary.perturb(data, labels_)
        fn='adv'
    else:
        adv_untargeted=data
        fn='norm'
    print (adv_untargeted.size())

    pred_n=test(args, model, device, test_loader_small, hookF, adv_untargeted)

    activity_layer={}
    for ttin in range(4):
        layer_sel_=ttin
        act_map=Associations[layer_sel_]
        

        roV=intermediate_output[layer_sel_]

        sel=Associations[layer_sel_]
        sel=sel.numpy()

        wm=torch.load('wm_'+str(layer_sel_)+'.pt',map_location=lambda storage, loc: storage)
        
        fp=open('labels_'+str(layer_sel_)+'.json') # labels for wm i.e., the labels of the test set.
        label=json.load(fp)
        fp.close()
        cog=CogMem_load(wm,label) 



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
        #print ('sel shape',sel.shape)
        #print (cog.image.size())
        #print ('pred',pred.size())
        temp_vec=[]       
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
            temp_v=np.exp(temp_v)
            temp_v=temp_v/np.sum(temp_v)
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
                    temp.append(np.dot(pred[zin],post[zin]))
                #temp=np.array(temp)
                layer_corr[str(xin)+'_'+str(yin)]=temp
                

    fp=open('layer-'+fn+'_'+'.json','w')
    json.dump(layer_corr,fp)


    
                    
                       
    
   
        
      
        
        

if __name__ == '__main__':
    main()
    del intermediate_output

