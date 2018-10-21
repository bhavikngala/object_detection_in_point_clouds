import torch

rootDir = './../data/KITTI_BEV'
trainRootDir = './../data/preprocessed/train'

gridConfig = {
	'x':(0, 70),
	'y':(-40, 40),
	'z':(-2.5, 1),
	'res':0.1
}

objtype = 'car'

# res_block_layers = list if number of channels in the first conv layer of each res_block
# up_sample_layers = list of tuple of number of channels input deconv and conv layers

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]

# training parameters
learningRate = 1e-2
epochs = 200

# gamma, alpha, epsilon for focal loss
gamma = 2
alpha = 0.25
epsilon = 1e-5

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

# calibration dir
calTrain = './../data_object_calib/training/calib'
calTest = './../data_object_calib/testing/calib'

# string for log
logString1 = 'epoch: {:05d} | batch:{:05d} | cla loss: {:.4f} | loc loss: {:.4f} | total loss: {:.4f} | lt : {:.4f} | bt : {:.4f} \n\n'
logString2 = 'epoch: {:05d} | batch:{:05d} | cla loss: {:.4f} | loc loss: None | total loss: {:.4f} | lt : {:.4f} | bt : {:.4f} \n\n'
logString3 = 'epoch: {:05d} | batch:{:05d} | cla loss: None | loc loss: None| total loss: None | lt : {:.4f} | bt : {:.4f} \n\n'
# logString = Template('epoch: $e | cla loss: $cl | loc loss: $ll | total loss: $tl \n')

numBatchInQueue = 4
batchSize = 3