#! /bin/bash

for s in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.18
do
python gen_adv.py --eps $s
done

python gen_adv.py --norm
