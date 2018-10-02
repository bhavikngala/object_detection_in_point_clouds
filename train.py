import torch
import torch.nn.functional as F
import torch.optim as optim
import os

from networks.networks import PointCloudDetector as HawkEye
from datautils.dataloader import *
import config as cnf
from lossutils import computeLoss
import misc

torch.manual_seed(0)

# data loaders
train_loader, vali_loader, test_loader = \
	lidarDatasetLoader(cnf.rootDir, cnf.batchSize, cnf.gridConfig, cnf.objtype)

# create detector object and intialize weights
hawkEye = HawkEye(cnf.res_block_layers, cnf.up_sample_layers).to(cnf.device)
hawkEye.apply(misc.weights_init)

# network optimization method
optimizer = optim.Adam(hawkEye.parameters(), lr=cnf.learningRate)

def train(epoch):
	hawkEye.train()

	for batchId, (data, target, filenames) in enumerate(train_loader):
		# move data to GPU
		data = data.to(cnf.device)
		target = target.to(cnf.device)

		# empty the gradient buffer
		optimizer.zero_grad()

		# pass data through network and predict
		cla, loc = hawkEye(data)

		# compute loss, gradient, and optimize
		claLoss, locLoss = computeLoss(cla, loc, target)
		trainLoss = claLoss + locLoss
		trainLoss.backward()
		optimizer.step()

		# TODO: mAP

		# save the results, loss in a file
		misc.savebatchOutput(cla, loc, filenames, cnf.trainOutputDir, epoch)
		misc.writeToFile(cnf.trainlog, cnf.logString.format(epoch, claLoss, locLoss, trainLoss))

def validation(epoch):
	hawkEye.eval()

	for batchId, (data, target, filenames) in enumerate(vali_loader):
		# move data to GPU
		data = data.to(device)
		target = target.to(device)

		# pass data through network and predict
		cla, loc = hawkEye(data)
		claLoss, locLoss = computeLoss(cla, loc, target, gamma)
		valiLoss = claLoss + locLoss

		# TODO mAP

		# save the results, loss in a file
		misc.savebatchOutput(cla, loc, filenames, cnf.valiOutputDir, epoch)
		misc.writeToFile(cnf.valilog, cnf.logString.format(epoch, claLoss, locLoss, valiLoss))

if __name__ == '__main__':
	# load model file if present
	if os.path.isfile(cnf.model_file):
		hawkEye.load_state_dict(torch.load(cnf.model_file,
			map_location=lambda storage, loc: storage))

	for i in range(cnf.epochs):
		train(epoch)
		validation(epoch)

		if (epoch+1)%10 == 0:
			torch.save(hawkEye.state_dict(), cnf.model_file)