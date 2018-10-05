# Object detection in point clouds
This project aims are detecting and localizing objects in point clouds captured from LIDAR data.
This project is uses the network in the following paper as a base model:</br>
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