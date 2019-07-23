import json
import numpy as np
import pylab


fp=open('layer-'+'adv_.json','r')
adv_data=json.load(fp)
fp.close()

fp=open('layer-'+'norm_.json','r')
norm_data=json.load(fp)
fp.close()

norm=[]
adv=[]

norm_group=[]
adv_group=[]
pre=['0','1']
post=['2','3']

diff=[]
for pr in pre:
    for po in post:
        if pr+'_'+po in norm_data:
            norm.append(norm_data[pr+'_'+po])
            adv.append(adv_data[pr+'_'+po])
            diff.extend(list(np.array(norm_data[pr+'_'+po])-np.array(adv_data[pr+'_'+po])))
    
           
norm=np.array(norm)
adv=np.array(adv)
diff=np.array(diff)

norm_group=np.mean(norm,0)
adv_group=np.mean(adv,0)

norm_h,norm_edges=np.histogram(norm_group)
adv_h,adv_edges=np.histogram(adv_group)
pylab.figure(1)
pylab.plot(norm_h,label='norm')
pylab.plot(adv_h,label='adv')
pylab.legend()

pylab.figure(2)
pylab.plot(diff)

print (len(np.where(diff<0)[0]), 'out of', len(diff))

pylab.show()
    
    
  


