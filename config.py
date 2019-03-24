import torch
import numpy as np

rootDir = './../data/tiny_set'
logDir = './runs'
logJSONFilename = './loss/logs.json'
trainSplitFile = './datautils/set_split_files/train.txt'
fullTrainFile = './datautils/set_split_files/trainval.txt'
valSplitFile = './datautils/set_split_files/val.txt'
testFile = './datautils/set_split_files/test.txt'

targetCord = 'velo'

imgHeight, imgWidth = 376, 1241

gridConfig = {
	'x':(0, 70.4),
	'y':(-40, 40),
	'z':(-2.5, 1),
	'res':0.1
}

x_min = gridConfig['x'][0]
x_max = gridConfig['x'][1]
y_min = gridConfig['y'][0]
y_max = gridConfig['y'][1]
z_min = gridConfig['z'][0]
z_max = gridConfig['z'][1]

x_axis = np.arange(x_min, x_max, gridConfig['res'])
y_axis = np.arange(y_min, y_max, gridConfig['res'])

x_mean, x_std = x_axis.mean(), x_axis.std()
y_mean, y_std = y_axis.mean(), y_axis.std()

d_x_min = -1.0
d_x_max =  1.0
d_y_min = -1.0
d_y_max =  1.0

lgrid = x_max-x_min
wgrid = y_max-y_min

diagx = np.sqrt(0.4**2 + 0.4**2)
diagy = np.sqrt(0.4**2 + 0.4**2)
la = 0.4
wa = 0.4

d_xy = np.sqrt((x_max-x_min)**2 + (y_max-y_min)**2)

in_channels = int((z_max-z_min)/gridConfig['res']+1)

downsamplingFactor = 4
r = int((y_max-y_min)/(gridConfig['res']*downsamplingFactor))
c = int((x_max-x_min)/(gridConfig['res']*downsamplingFactor))

objtype = 'car'

carPIXORIgnoreBoundaryMean = torch.tensor([-0.0004,  0.0089,  0.2001, -0.2015,  1.3619,  0.4839], dtype=torch.float32)
carPIXORIgnoreBoundarySTD = torch.tensor([0.4530, 0.8915, 0.4263, 0.3093, 0.1131, 0.0650], dtype=torch.float32)

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]
deconv = [(1, 2, 1, 1), # upsamole block 1
		  (1, 2, 1, 1)] # upsample block 2

pretrainCla = True
cycleLearn  = False
# training parameters
lr = 1e-4   # learning rate without step
slr = 1e-2  # step learning rate
lrDecay = 0.1 # learning rate decay
milestones = [20, 30] # milestone for pixor
momentum = 0.9
decay = 0.0001 # weight decay parameter
epochs = 40
stepSize = epochs//2
lrRange = [1e-4, 1e-2]
momentumRange = [0.85, 0.9]
lrRange2 = [1e-5, 1e-4]
momentumRange2 = [0.9, 0.9]

# balancing classification and regression losses
alpha1 = 1.0
beta1 = 0.1

# gamma, alpha, epsilon for focal loss
gamma = 2
alpha = 0.25
epsilon = 1e-5

# select gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# output directories for train, validation, and test outputs
trainOutputDir = './output/train'
valiOutputDir = './output/val'
testOutputDir = './output/test'


# calibration dir
calTrain = './../data_object_calib/training/calib'
calTest = './../data_object_calib/testing/calib'

# left color images dir
leftColorTrain = './../data/left_color_images/data_object_image_2/training/image_2'
leftColorTest = './../data/left_color_images/data_object_image_2/testing/image_2'

batchSize = 4
accumulationSteps = 4.0
clip_value = 1

# threshold for NMS
iouThreshold = 0.1