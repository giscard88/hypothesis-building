#! /bin/bash

for s in 0 1 2 3 4 5 6
do

for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 3.5 4.0
do

python gen_layer_correlation.py --set $s --eps $i

done
python gen_layer_correlation.py --set $s --norm
done



