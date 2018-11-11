import torch
import numpy as np

rootDir = './../data/KITTI_BEV'
trainRootDir = './../data/preprocessed/train'

gridConfig = {
	'x':(0, 70),
	'y':(-40, 40),
	'z':(-2.5, 1),
	'res':0.1
}

Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
	])
# cal mean from train set
R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
])
P2 = np.array([[719.787081,         0., 608.463003,    44.9538775],
               [        0., 719.787081, 174.545111,     0.1066855],
               [        0.,         0.,         1., 3.0106472e-03],
			   [0., 0., 0., 0]])
R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)

objtype = 'car'

carTargetMean = np.array([[ 0.8216354, 0.08494052, 28.304243, 2.4187818, 1.3506572, 0.48570955]])
carSTD = np.array([[16.32046613, 8.3399593, 0.36412882, 0.13669496, 0.10216206, 0.42591674, 0.63424664]])

# res_block_layers = list if number of channels in the first conv layer of each res_block
# up_sample_layers = list of tuple of number of channels input deconv and conv layers
# deconv = tuplr of (dilation, stride, padding, output_padding) for deconvolution in upsampling

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]
deconv = [(1, 2, (1, 1), (1, 1)), # upsamole block 1
		  (1, 2, (1, 1), (1, 0))] # upsample block 2

# training parameters
lr = 1e-4   # learning rate without step
slr = 1e-2  # step learning rate
epochs = 200

# gamma, alpha, epsilon for focal loss
gamma = 2
alpha = 0.25
epsilon = 1e-5

# select gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# posLabel = torch.Tensor([1.0]).to(device)
# negLabel = torch.Tensor([0.0]).to(device)

# filename of saved model
model_file = './models/hawkEye.pth'

# output directories for train, validation, and test outputs
trainOutputDir = './output/train'
valiOutputDir = './output/val'
testOutputDir = './output/test'

# train, validation, test loss log file
trainlog = './loss/train.txt'
trainlog2 = './loss/etime.txt'
vallog = './loss/vali.txt'
testlog = './loss/test.txt'
errorlog = './loss/error.txt'

# calibration dir
calTrain = './../data_object_calib/training/calib'
calTest = './../data_object_calib/testing/calib'

# string for log
logString1 = 'epoch: {:03d} | batch:{:04d} | cla loss: {:.8f} | loc loss: {:.8f} | total loss: {:.8f} | PS : {:07d} | NS : {:07d} | iou : {:.4f} | mc : {:.4f} | lt : {:.4f} | bt : {:.4f} \n\n'
logString2 = 'epoch: {:03d} | batch:{:04d} | cla loss: {:.8f} | loc loss: None | total loss: {:.8f} | PS : {:07d} | NS : {:07d} | iou : None | mc : None | lt : {:.4f} | bt : {:.4f} \n\n'
logString3 = 'epoch: {:03d} | batch:{:04d} | cla loss: None | loc loss: None| total loss: None | PS : {:07d} | NS : {:07d} | iou : None | mc : None | lt : {:.4f} | bt : {:.4f} \n\n'
# logString = Template('epoch: $e | cla loss: $cl | loc loss: $ll | total loss: $tl \n')

batchSize = 3
accumulationSteps = 4.0