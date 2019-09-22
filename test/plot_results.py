import argparse
import json
import numpy as np
import pylab
import os
from sklearn.metrics import roc_auc_score, roc_curve


parser = argparse.ArgumentParser(description='gen adversarial examples via advertorch')


parser.add_argument('--set', type=str, default='0',
                        help='set of layers (default: 0)')
parser.add_argument('--threshold', type=str, default='0.1',
                        help='threshokd (default: 0.1)')
args = parser.parse_args()



fp=open('correlations/layer-adv_'+args.set+'-'+args.threshold+'.json','r')
adv_data=json.load(fp)
fp.close()

fp=open('correlations/layer-norm_'+args.set+'.json','r')
norm_data=json.load(fp)
fp.close()

norm=[]
adv=[]

norm_group=[]
adv_group=[]
pre=['1','2','3']
post=['2','3','4']

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

if os.path.exists("figs"):
    pass
else:
    os.mkdir("figs")


    
norm_h,norm_edges=np.histogram(norm_group)
adv_h,adv_edges=np.histogram(adv_group)
pylab.figure(1)
pylab.plot(norm_edges[1:],norm_h,label='norm')
pylab.plot(adv_edges[1:],adv_h,label='adv')
pylab.legend()



pylab.savefig('consistency_'+args.set+'_'+args.threshold+'.eps')
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

pylab.figure(2)
pylab.plot(fpr, tpr, lw=2)
pylab.savefig('roc_'+args.set+'_'+args.threshold+'.eps')



pylab.show()
    
    
  


