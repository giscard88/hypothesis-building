import json
import numpy as np
import pylab
from sklearn.metrics import roc_auc_score, roc_curve

fp=open('correlations/layer-adv_1-1.0.json','r')
adv_data=json.load(fp)
fp.close()

fp=open('correlations/layer-norm_1.json','r')
norm_data=json.load(fp)
fp.close()

norm=[]
adv=[]

norm_group=[]
adv_group=[]
pre=['0','1','2']
post=['2','3']

diff=[]
for pr in pre:
    for po in post:
        if pr==po:
            pass
        else:

            if pr+'_'+po in norm_data:
                norm.append(norm_data[pr+'_'+po])
                adv.append(adv_data[pr+'_'+po])
                diff.extend(list(np.array(norm_data[pr+'_'+po])-np.array(adv_data[pr+'_'+po])))
    
           
norm=np.array(norm)
adv=np.array(adv)
#diff=np.array(diff)

norm_group=np.mean(norm,0)
adv_group=np.mean(adv,0)
y_true=[]
y_pred=[]
for xi, xin in enumerate(adv_group):
    y_pred.append(xin)
    y_true.append(0)
    y_pred.append(norm_group[xi])
    y_true.append(1)

y_pred=np.array(y_pred)
y_true=np.array(y_true)

print ('roc',roc_auc_score(y_true, y_pred))
    
norm_h,norm_edges=np.histogram(norm_group)
adv_h,adv_edges=np.histogram(adv_group)
pylab.figure(1)
pylab.plot(norm_edges[1:],norm_h,label='norm')
pylab.plot(adv_edges[1:],adv_h,label='adv')
pylab.legend()

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

pylab.figure(2)
pylab.plot(fpr, tpr, lw=2)




pylab.show()
    
    
  


