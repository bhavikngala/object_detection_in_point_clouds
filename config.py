import torch
import numpy as np

rootDir = './../data/KITTI_BEV'
trainRootDir = './../data/preprocessed/train'

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

d_xy = np.sqrt((x_max-x_min)**2 + (y_max-y_min)**2)

in_channels = (z_max-z_min)/gridConfig['res']+1

downsamplingFactor = 4
r = int((y_max-y_min)/(gridConfig['res']*ds))
c = int((x_max-x_min)/(gridConfig['res']*ds))

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
			   [0., 0., 0., 0]
])
R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)

objtype = 'car'

# without log(l) and log(w)
carMean2 = np.array([[0, 0, 28.304243, 2.4187818, 3.8839872, 1.6286187]], dtype=np.float32)
carSTD2 = np.array([[1, 1, 16.32045, 8.339981, 0.4259141, 0.10216219]], dtype=np.float32)
zoom03Mean = np.array([[28.136889, 2.207453, 28.097183, 2.5897021, 28.471579, 2.6301286, 28.511364, 2.247868]], dtype=np.float32)
zoom03STD = np.array([[16.30122, 8.321556, 16.322311, 8.354368, 16.341236, 8.359552, 16.319061, 8.329018]], dtype=np.float32)
zoom12Mean = np.array([[27.634792, 1.5734423, 27.475872, 3.102446, 28.973452, 3.2641282, 29.13241, 1.7351091]], dtype=np.float32)
zoom12STD = np.array([[16.25314, 8.273551, 16.331024, 8.418133, 16.41331, 8.425362, 16.318035, 8.317054]], dtype=np.float32)

carMeanLogWL = np.array([[0, 0, 28.304243, 2.4187818, 1.3506572, 0.48570955]], dtype=np.float32)
carSTDLogWL = np.array([[1, 1, 16.32045, 8.339981, 0.1121646, 0.06327728]], dtype=np.float32)

carMean = carMeanLogWL
carSTD = carSTDLogWL

# res_block_layers = list if number of channels in the first conv layer of each res_block
# up_sample_layers = list of tuple of number of channels input deconv and conv layers
# deconv = tuplr of (dilation, stride, padding, output_padding) for deconvolution in upsampling

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]
deconv = [(1, 2, 1, 1), # upsamole block 1
		  (1, 2, 1, 1)] # upsample block 2

# training parameters
lr = 1e-4   # learning rate without step
slr = 1e-2  # step learning rate
milestones = [40, 45] # milestone for pixor
momentum = 0.9
decay = 0.0001 # weight decay parameter
epochs = 50

# balancing pos-neg samples
alpha1 = 1.5
beta1 = 1.0

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
gradNormlog = './loss/gnorm.txt'

# calibration dir
calTrain = './../data_object_calib/training/calib'
calTest = './../data_object_calib/testing/calib'

# string for log
logString1 = 'epoch: [{:04d}/{:03d}] | cl: {:.8f} | nsl: {:.8f} | psl: {:.8f} | ll: {:.8f} | tl: {:.8f} | PS: [{:07d}/{:07d}] | md: {:.4f} | mc: {:.4f} | oamc: {:.4f} | lt: {:.4f} | bt: {:.4f} \n\n'
logString2 = 'epoch: [{:04d}/{:03d}] | cl: {:.8f} | nsl: {:.8f} | psl: -.-------- | ll: -.-------- | tl: {:.8f} | PS: [{:07d}/{:07d}] | md: -.---- | mc: -.---- | oamc: {:.4f} | lt: {:.4f} | bt: {:.4f} \n\n'
logString3 = 'epoch: [{:04d}/{:03d}] | cl: -.-------- | nsl: -.-------- | psl: -.-------- | ll: -.--------| tl: -.-------- | PS: [{:07d}/{:07d}] | md: -.---- | mc: -.---- | oamc: -.---- | lt: {:.4f} | bt: {:.4f} \n\n'
normLogString = 'epoch: [{:04d}/{:03d}] | grad norm: {:.8f} | weight norm: {:.8f} \n\n'

batchSize = 4
accumulationSteps = 4.0
