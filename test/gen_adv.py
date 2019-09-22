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
from resnet import *
from advertorch.attacks import LinfPGDAttack



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='gen adversarial examples via advertorch')


    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--eps', default=0.3,
                        help='eps for LinfPGDAttack')

    parser.add_argument('--norm', action='store_true', default=False,
                        help='adversarial?')
  
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
        batch_size=200, shuffle=False,**kwargs)

    model=resnet44()
    model.to(device)

    checkpoint = torch.load('pretrained_models/resnet44.th',map_location=lambda storage, loc: storage)
    states=checkpoint['state_dict']
    del checkpoint
    param_dict={}


    for pr in states:
        param_dict[pr[7:]]=states[pr]

    model.load_state_dict(param_dict)
    
    adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=float(args.eps),
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

    for data, target in test_loader:
        break
    labels_=target
    data=data.to(device)
    labels_=labels_.to(device)

    if os.path.exists("adversarial_examples"):
        pass
    else:
        os.mkdir("adversarial_examples")

    if args.norm:    
        torch.save(data, 'adversarial_examples/norm.pt')
        torch.save(labels_, 'adversarial_examples/norm_label_.pt') 
    else:
        adv_untargeted = adversary.perturb(data, labels_)
        torch.save(adv_untargeted, 'adversarial_examples/adv_'+str(args.eps)+'.pt')
        torch.save(labels_, 'adversarial_examples/adv_label_'+str(args.eps)+'.pt')
       
    
    
       
if __name__ == '__main__':
    main()
    del internal




