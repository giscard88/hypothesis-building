#! /bin/bash

for s in 0 1 2 3 4 5 6
do

for i in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do

python gen_layer_correlation.py --set $s --eps $i

done
python gen_layer_correlation.py --set $s --norm
done



