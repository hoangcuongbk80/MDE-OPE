
## Geometry-Aware Object Pose Estimation Using Color Images

### Installation
- Install CUDA 10.1 / 10.2
- Set up python3 environment from requirement.txt:
  ```shell
  pip3 install -r requirement.txt 
  ```
- Install [apex](https://github.com/NVIDIA/apex):
  ```shell
  git clone https://github.com/NVIDIA/apex
  cd apex
  export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001
  pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  cd ..
  ```
### Datasets

- **LineMOD:** Download the preprocessed LineMOD dataset from [onedrive link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhebk_connect_ust_hk/ETW6iYHDbo1OsIbNJbyNBkABF7uJsuerB6c0pAiiIv6AHw?e=eXM1UE) or [google drive link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (refer from [DenseFusion](https://github.com/j96w/DenseFusion)). Unzip it and link the unzipped ``Linemod_preprocessed/`` to ``datasets/linemod/Linemod_preprocessed``:
  ```shell
  ln -s path_to_unzipped_Linemod_preprocessed mde6d/dataset/linemod/
  ```
  Generate rendered and fused data following [raster_triangle](https://github.com/ethnhe/raster_triangle).

- **YCB-Video:** Download the YCB-Video Dataset from [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/). Unzip it and link the unzipped```YCB_Video_Dataset``` to ```datasets/ycb/YCB_Video_Dataset```:

  ```shell
  ln -s path_to_unzipped_YCB_Video_Dataset mde6d/datasets/ycb/
  ```

## Training and evaluating

### Training on the LineMOD Dataset
- Train the model for the target object. Take object ape for example:
  ```shell
  # commands in train_lm.sh
  n_gpu=8
  cls='ape'
  python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls
  ```
  The trained checkpoints are stored in ``train_log/linemod/checkpoints/{cls}/``, ``train_log/linemod/checkpoints/ape/`` in this example.

### Evaluating on the LineMOD Dataset
- Start evaluation by:
  ```shell
  # commands in test_lm.sh
  cls='ape'
  tst_mdl="./linemod_pretrained/mde6d_${cls}_best.pth.tar"
  python3 -m torch.distributed.launch --nproc_per_node=1 train_lm.py --gpu '0' --cls $cls -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
  ```

### Demo/visualizaion on the LineMOD Dataset
- After training your models, you can start the demo by:
  ```shell
  # commands in demo_lm.sh
  cls='ape'
  tst_mdl=train_log/linemod/checkpoints/${cls}/mde6d_${cls}_best.pth.tar
  python3 -m demo -dataset linemod -checkpoint $tst_mdl -cls $cls -show
  ```
  The visualization results will be stored in ``train_log/linemod/eval_results/{cls}/pose_vis``

### Training on the YCB-Video Dataset
- Start training on the YCB-Video Dataset by:
  ```shell
  # commands in train_ycb.sh
  n_gpu=8  # number of gpu to use
  python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu
  ```
  The trained model checkpoints are stored in ``train_log/ycb/checkpoints/``


### Evaluating on the YCB-Video Dataset
- Start evaluating by:
  ```shell
  # commands in test_ycb.sh
  tst_mdl=train_log/ycb/checkpoints/mde6d_best.pth.tar  # checkpoint to test.
  python3 -m torch.distributed.launch --nproc_per_node=1 train_ycb.py --gpu '0' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
  ```

