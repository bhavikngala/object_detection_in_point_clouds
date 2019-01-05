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

### Train script command line arguments
```
python train.py [options] [-f] [-m]

	-f, --model-file, type = string, description = Used to specify model file name used for checkpointing and loading pretrained model if required, required = True
	-m, --multi-gpu, type = boolean, description = Set to train model on multiple GPUs, default = False
```

### directory structure
--Root dir
----code 				[contains all the code]
------models			[contains saved model files]
------output 			[output labels from the validateNetwork.py script are stored here]
--------[model_name]	[the folder name should be same as specified in validatenetwork.py script]
----------data 			[folder where labels.txt will be saved from validateNetwork.py script]
----data 				[directory where train/val and test data is stored]
------training 			[training (train+val) data]
--------velodyne 		[velodyne point cloud data files]
----------label_2 		[labels.txt]

### Steps to train
1. Install pre-requisites
2. Install the environment: `conda env create -f environment.yml`
3. Run train.py: `python train.py -f ./models/[model name].pth -m &`

### Steps to validate network
1. To validate the network the directory structure should be as specified above
2. Change the root directory name for data folder in config.py script, variable name to change is `rootDir`. The path is relative to the code directory
3. In the validateNetwork.py script change the output directory where the labels generated for the validation set will be saved, it is hardcored for now
4. Open command line, change directory to code directory
5. Activate conda environment
6. Run `python validateNetwork.py -f [path to model file] &` to generate object localization on validation set
7. Predicted locations will be saved in `code/output/[model name]/data`