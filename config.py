import torch

rootDir = './../data/KITTI_BEV'
batchSize = 2

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
learningRate = 1e-4
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
valilog = './loss/vali.txt'
testlog = './loss/test.txt'

# string for log
logString = 'epoch: {:05d} | cla loss: {:.4f} | loc loss: {:.4f} | total loss: {:.4f} \n'