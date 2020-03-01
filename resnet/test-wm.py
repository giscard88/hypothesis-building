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

for xin in range(5):
    data=torch.load('wm_'+str(xin)+'.pt',map_location=lambda storage, loc: storage)
    print (xin, data.size())
