import torch
# from string import Template

rootDir = './../data/KITTI_BEV'
batchSize = 3

gridConfig = {
	'x':(0, 70),
	'y':(-40, 40),
	'z':(-2.5, 1),
	'res':0.1
}

objtype = 'Car'

# res_block_layers = list if number of channels in the first conv layer of each res_block
# up_sample_layers = list of tuple of number of channels input deconv and conv layers

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]

# training parameters
learningRate = 1e-2
epochs = 200

# gamma for focal loss
gamma = 2

# select gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
posLabel = torch.Tensor([1.0]).to(device)
negLabel = torch.Tensor([0.0]).to(device)

# filename of saved model
model_file = './models/hawkEye.pth'

# output directories for train, validation, and test outputs
trainOutputDir = './output/train'
valiOutputDir = './output/val'
testOutputDir = './output/test'

# train, validation, test loss log file
trainlog = './loss/train.txt'
vallog = './loss/vali.txt'
testlog = './loss/test.txt'

# string for log
logString1 = 'epoch: {:05d} | batch:{:05d} | cla loss: {:.4f} | loc loss: {:.4f} | total loss: {:.4f} \n'
logString2 = 'epoch: {:05d} | batch:{:05d} | cla loss: {:.4f} | loc loss: None | total loss: {:.4f} \n'
logString3 = 'epoch: {:05d} | batch:{:05d} | cla loss: None | loc loss: None| total loss: None \n'
# logString = Template('epoch: $e | cla loss: $cl | loc loss: $ll | total loss: $tl \n')

# camera and lidar conversion matrices
MATRIX_P2 = ([
	[719.787081,    0.,            608.463003, 44.9538775],
	[0.,            719.787081,    174.545111, 0.1066855],
	[0.,            0.,            1.,         3.0106472e-03],
	[0.,            0.,            0.,         0]
])
MATRIX_T_VELO_2_CAM = ([
	[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
	[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
	[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
	[0, 0, 0, 1]
 ])
MATRIX_R_RECT_0 = ([
	[0.99992475, 0.00975976, -0.00734152, 0],
	[-0.0097913, 0.99994262, -0.00430371, 0],
	[0.00729911, 0.0043753, 0.99996319, 0],
	[0, 0, 0, 1]
])