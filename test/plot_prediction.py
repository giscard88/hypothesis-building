import numpy as np
import json
import pylab
import os
from collections import defaultdict


cwd=os.getcwd()
targetdir=cwd+'/confusion/'
target_layers=[0,2,3,4]
preds_=defaultdict(list)
for tl in target_layers:
    fp=open(targetdir+'prediction'+str(tl)+'.json','r')
    data=json.load(fp)
    
    for th in data:
        preds_[str(tl)+'_thresholds'].append(th)
        for arg in data[th]:
            preds_[str(tl)+'_'+arg].append(data[th][arg])
        
print (preds_)

pylab.figure(1,figsize=(20,10))
pylab.suptitle('consistent_pred')
for f,fi in enumerate(target_layers):
    pylab.subplot(2,2,f+1)
    pylab.plot(preds_[str(fi)+'_thresholds'], np.array(preds_[str(fi)+'_consist_pred'])/10000.0,'-o')
    pylab.title('layer:'+str(fi))
    pylab.xlabel('threshold')
    pylab.ylabel('correct answer (%)')
    pylab.savefig('consistent_pred.png')
    pylab.savefig('consistent_pred.eps')

pylab.figure(2,figsize=(20,10))
pylab.suptitle('cog_size')
for f,fi in enumerate(target_layers):
    pylab.subplot(2,2,f+1)
    yv=[]
    for y in preds_[str(fi)+'_cog_size']:
        yv.append(y[0])
    pylab.plot(preds_[str(fi)+'_thresholds'], yv,'-o')
    pylab.title('layer:'+str(fi))
    pylab.xlabel('threshold')
    pylab.ylabel('Library network size')

    pylab.savefig('cog_size.png')
    pylab.savefig('cog_size.eps')

pylab.figure(3,figsize=(20,10))
pylab.suptitle('max_pred')
for f,fi in enumerate(target_layers):
    pylab.subplot(2,2,f+1)
    pylab.plot(preds_[str(fi)+'_thresholds'], np.array(preds_[str(fi)+'_max_pred'])/10000.,'-o')
    pylab.title('layer:'+str(fi))
    pylab.xlabel('threshold')
    pylab.ylabel('correct answer (%)')
    pylab.savefig('max_prod.png')
    pylab.savefig('max_prod.eps')

pylab.show() 
