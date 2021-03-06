import numpy as np
import json
import pylab
import os
from collections import defaultdict


cwd=os.getcwd()
targetdir=cwd+'/confusion/'
target_layers=[0, 1, 2, 3]
preds_=defaultdict(list)

#thres_=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9] 
for tl in target_layers:
    fp=open(targetdir+'prediction'+str(tl)+'.json','r')
    data=json.load(fp)
    
    for th in data:
        preds_[str(tl)+'_thresholds'].append(th)

        for arg in data[th]:
            preds_[str(tl)+'_'+arg].append(data[th][arg])

    
  

pylab.figure(1)

for fi in target_layers:
    pylab.plot(preds_[str(fi)+'_thresholds'], np.array(preds_[str(fi)+'_consist_pred'])/10000.0,'-o',label='L'+str(fi))
    pylab.title('consistent_pred')
    pylab.xlabel('threshold')
    pylab.ylabel('correct answer (%)')
    pylab.legend()
    
    pylab.savefig('compact_consistent_pred.png')
    pylab.savefig('compact_consistent_pred.eps')




pylab.figure(2)

for f,fi in enumerate(target_layers):

    yv=[]
    for y in preds_[str(fi)+'_cog_size']:
        yv.append(y[0])
    pylab.semilogy(preds_[str(fi)+'_thresholds'], yv,'-o',label='L'+str(fi))
    pylab.title('cog_size')
    pylab.xlabel('threshold')
    pylab.ylabel('Library network size')
    pylab.legend()
    pylab.savefig('compact_cog_size.png')
    pylab.savefig('compact_cog_size.eps')

pylab.figure(3)

for f,fi in enumerate(target_layers):
    pylab.plot(preds_[str(fi)+'_thresholds'], np.array(preds_[str(fi)+'_max_pred'])/10000.,'-o',label='L'+str(fi))
    pylab.title('max_pred')
    pylab.xlabel('threshold')
    pylab.ylabel('correct answer (%)')
    pylab.legend()
    pylab.savefig('compact_max_prod.png')
    pylab.savefig('compact_max_prod.eps')


pylab.show() 
