import numpy as np
import json
import pylab
import os
from collections import defaultdict


cwd=os.getcwd()
targetdir=cwd+'/confusion/'
target_layers=[1,2,3,4]
preds_=defaultdict(list)
for tl in target_layers:
    fp=open(targetdir+'prediction'+str(tl)+'.json','r')
    data=json.load(fp)
    
    for th in data:
        preds_[str(tl)+'_thresholds'].append(th)
        for arg in data[th]:
            preds_[str(tl)+'_'+arg].append(data[th][arg])
        
print (preds_)

pylab.figure(1)
pylab.suptitle('consistent_pred')
for fi in range(4):
    pylab.subplot(2,2,fi+1)
    pylab.plot(preds_[str(fi+1)+'_thresholds'], preds_[str(fi+1)+'_consist_pred'])


pylab.figure(2)
pylab.suptitle('cog_size')
for fi in range(4):
    pylab.subplot(2,2,fi+1)
    yv=[]
    for y in preds_[str(fi+1)+'_cog_size']:
        yv.append(y[0])
    pylab.plot(preds_[str(fi+1)+'_thresholds'], yv)


pylab.figure(3)
pylab.suptitle('max_pred')
for fi in range(4):
    pylab.subplot(2,2,fi+1)
    pylab.plot(preds_[str(fi+1)+'_thresholds'], preds_[str(fi+1)+'_max_pred'])


pylab.show() 
