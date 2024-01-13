#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/mde6d_best.pth.tar  # checkpoint to test.
python3 -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu '0' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
