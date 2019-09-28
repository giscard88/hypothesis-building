import numpy as np
import json
import pylab
import os
from collections import defaultdict


cwd=os.getcwd()


fp=open('consistency.json','r')
data=json.load(fp)
fp.close()
layers=[]
thres=[]
for arg in data:
    pt=arg.find('_')
    layer=arg[:pt]
    th=arg[pt+1:]
    layers.append(layer)
    thres.append(th)

colors={'0':'-ro','1':'-go','2':'-bo','3':'-ko','4':'-co','5':'-mo','6':'-yo'}
thres=np.array(thres).astype(float)
thres=np.sort(thres)
idx=np.where(thres<=1.0)
thres=thres[idx]
display=[]
pylab.figure(1)
for la in layers:
    col=colors[str(la)]
    temp=np.zeros(len(thres))*np.nan
    for t, th in enumerate(thres):
        temp[t]=data[str(la)+'_'+str(th)]
    if str(la) not in display:
        pylab.plot(thres,temp,col,label='set'+str(la))
        display.append(str(la))
    else:
        pylab.plot(thres,temp,col)
pylab.legend()
pylab.savefig('consistency_set.eps')
pylab.show()
        
        
        
