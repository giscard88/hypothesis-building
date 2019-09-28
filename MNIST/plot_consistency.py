import argparse
import json
import numpy as np
import pylab
import os
from sklearn.metrics import roc_auc_score, roc_curve


parser = argparse.ArgumentParser(description='gen adversarial examples via advertorch')




def gen_figs(sel_set, sel_threshold):
    fp=open('correlations/layer-adv_'+sel_set+'-'+sel_threshold+'.json','r')
    adv_data=json.load(fp)
    fp.close()

    fp=open('correlations/layer-norm_'+sel_set+'.json','r')
    norm_data=json.load(fp)
    fp.close()

    norm=[]
    adv=[]

    norm_group=[]
    adv_group=[]
    pre=['0','1','2','3']
    post=['0','1','2','3']

    diff=[]
    for pr in pre:
        for po in post:
            if pr>=po:
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



    if os.path.exists("figs"):
        pass
    else:
        os.mkdir("figs")


    
    norm_h,norm_edges=np.histogram(norm_group,20)
    adv_h,adv_edges=np.histogram(adv_group,20)
    pylab.figure(1)
    pylab.plot(norm_edges[1:],norm_h,label='norm')
    pylab.plot(adv_edges[1:],adv_h,label='adv')
    pylab.legend()

    pylab.savefig('figs/consistency_'+sel_set+'_'+sel_threshold+'.eps')
    pylab.savefig('figs/consistency_'+sel_set+'_'+sel_threshold+'.png')
    pylab.close()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    pylab.figure(2)
    pylab.plot(fpr, tpr, lw=2)
    pylab.savefig('figs/roc_'+sel_set+'_'+sel_threshold+'.eps')
    pylab.savefig('figs/roc_'+sel_set+'_'+sel_threshold+'.png')
    pylab.close()
    return roc_auc_score(y_true, y_pred)

def main():
    roc_list={}
    for s in [0, 1, 2, 3, 4, 5, 6]:
        sel_set=str(s)        
        for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]:
            sel_threshold=str(i)
            roc_=gen_figs(sel_set,sel_threshold)
            roc_list[sel_set+'_'+sel_threshold]=roc_
    print (roc_list)
    fp=open('consistency.json','w')
    json.dump(roc_list,fp)
    fp.close()

if __name__ == '__main__':
    main()

        



    
    
  


