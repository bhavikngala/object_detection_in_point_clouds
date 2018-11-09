# Object detection in point clouds
This project aims at detecting and localizing objects in point clouds captured from LIDAR sensor.
This project uses the network in the following paper as a base model:</br>
Bin Yang, Wenjie Luo, Raquel Urtasun. <b>PIXOR: Real-time 3D Object Detection from Point Clouds.</b><i> In proceedings in CVPR 2018, pages 7652-7660.</i>

### Requirements
Pre-requisites:
1. Conda
2. CuDNN
3. Cuda 9.1

Requirements
1. Python3
2. Pytorch 0.4.0

### Steps to run
1. Install pre-requisites
2. Install the environment: `conda env create -f environment.yml`
3. Run train.py: `python train.py`

```
python train.py [options] [--step-lr] [--aug-data] [-f] [-r] [-p] [-v] [-e] [--aug-scheme] [-m]

	--step-lr, type = boolean, description = If set then step </br>learning rates is used, default = False
	--aug-data type = boolean, description = Used to turn data  </br>augmentation on/off, default = False
	-f, --model-file, type = string, description = Used to specify </br>model file name, defualt = None 
	-r, --root-dir, type = string, description = Used to specify </br>root directory for data, default = None
	-p, --pixor, type = boolean, description = Used when pixor data </br>augmentation method is to be used, default = False
	-v, --voxelnet, type = boolean, description = Used when voxelnet </br>data augmentation method is to be used, default = False
	-e, --epochs, type = int, description = Used to specify number </br>of epochs, default=None
	--aug_scheme, type = string, description = pixor of voxelnet </br>augmentation scheme, set internally depending on -p -v </br>flags, default=None
	-m, --multi-gpu, type = boolean, description = Set to train </br>model on multiple GPUs, default = False
```