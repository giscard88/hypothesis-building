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

    model=Net()
    model.to(device)
    checkpoint = torch.load('mnist_cnn.pt',map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)


    
    hookF=[Hook(layer [1]) for layer in list(model._modules.items())]
    test(args, model, device, train_loader, hookF)

    data_wm=[]
    Associations=[]
    for xin in [3]:
        wm=torch.load('wm_'+str(xin)+'.pt',map_location=lambda storage, loc: storage)
        data_wm.append(wm)
        fp=open('labels_'+str(xin)+'.json') # labels for wm i.e., the labels of the test set.
        label=json.load(fp)
        fp.close()
        cog=CogMem_load(wm,label) 
        roV=intermediate_output[xin]
        for data, target in train_loader:
            labels_=target
        cog.foward(roV) # estimate the images of train set to make associations
        temp=make_association(cog.image,labels_,device)
        print (temp.map.size())
        Associations.append(temp)
        
    
    # Let's test how this association is predictive of the test set
    
    
    temp=Associations[0]
    act_map=temp.map

    correlations=np.zeros((10,10))

    #act_map=act_map.numpy()
    test(args, model, device, test_loader, hookF)
    #correlations=np.zeros((10,10))
    roV=intermediate_output[3]
    for data, target in test_loader:
        labels_=target
        cog.foward(roV)
        pred=cog.pred.long()
    #pred=cog.pred.long().cpu().numpy()
    
    for xi, xin in enumerate(pred[:20]):
        cls=xin.item()
        
        v2=cog.image[:,xi]
        for yin in [cls]:
            v1=act_map[:,yin]
            print (cls,yin)
            print (torch.dot(v1,v2)/(v1.norm()*v2.norm()))
            #print (v2[:10])

    
    #print (cog.image.size())
    #print (cog.image[:10,1]) 

    #       correlations[xin,yin]=correlations[xin,yin]+torch.dot(v1,v2)
    #correlations=correlations
    #np.savetxt('test.txt',correlations)

           
        
        
      
        
        

if __name__ == '__main__':
    main()


