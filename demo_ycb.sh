#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/mde6d_best.pth.tar
python3 -m demo -checkpoint $tst_mdl -dataset ycb
