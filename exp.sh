#!/bin/sh

python train.py --dataname instagram --pseudospx --modelname scn --earlystp --genmasks
python train.py --dataname instagram --pseudospx --modelname sccn --earlystp