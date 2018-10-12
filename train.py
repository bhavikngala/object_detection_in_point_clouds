import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import time

from networks.networks import PointCloudDetector as HawkEye
from datautils.dataloader import *
import config as cnf
from lossUtils import computeLoss
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

	for batchId, batch_data in enumerate(train_loader):
		data, target, filenames, zoom0_3, zoom1_2 = batch_data
		# move data to GPU
		data = data.to(cnf.device)
		target = [t.to(cnf.device, non_blocking=True) for t in target]
		zoom1_2 = zoom1_2.to(cnf.device, non_blocking=True)
		zoom0_3 = zoom0_3.to(cnf.device, non_blocking=True)

		# empty the gradient buffer
		optimizer.zero_grad()

		# pass data through network and predict
		cla, loc = hawkEye(data)
		print('epoch:', epoch)

		# compute loss, gradient, and optimize
		claLoss, locLoss = computeLoss(cla, loc, target, zoom0_3, zoom1_2)
		trainLoss = claLoss + locLoss
		trainLoss.backward()
		optimizer.step()

		# TODO: mAP

		# save the results, loss in a file
		misc.savebatchOutput(cla, loc, filenames, cnf.trainOutputDir, epoch)
		misc.savebatchTarget(target, filenames, cnf.trainOutputDir, epoch)
		misc.writeToFile(cnf.trainlog, cnf.logString.format(epoch, claLoss.item(), locLoss.item(), trainLoss.item()))
		# print('train', cnf.logString.format(epoch, claLoss.item(), locLoss.item(), trainLoss.item()))

def validation(epoch):
	hawkEye.eval()

	for batchId, batch_data in enumerate(vali_loader):
		data, target, filenames, zoom0_3, zoom1_2 = batch_data

		# move data to GPU
		data = data.to(cnf.device)
		target = [t.to(cnf.device, non_blocking=True) for t in target]
		zoom1_2 = zoom1_2.to(cnf.device, non_blocking=True)
		zoom0_3 = zoom0_3.to(cnf.device, non_blocking=True)

		# pass data through network and predict
		cla, loc = hawkEye(data)
		print('epoch:', epoch)

		claLoss, locLoss = computeLoss(cla, loc, target, zoom0_3, zoom1_2)
		valiLoss = claLoss + locLoss

		# TODO mAP

		# save the results, loss in a file
		misc.savebatchOutput(cla, loc, filenames, cnf.valiOutputDir, epoch)
		misc.savebatchTarget(target, filenames, cnf.valiOutputDir, epoch)
		misc.writeToFile(cnf.valilog, cnf.logString.format(epoch, claLoss.item(), locLoss.item(), valiLoss.item()))
		# print('val', cnf.logString.format(epoch, claLoss.item(), locLoss.item(), valiLoss.item()))

if __name__ == '__main__':
	# current_milli_time = lambda: time.time()*1000
	# start = current_milli_time()

	# load model file if present
	if os.path.isfile(cnf.model_file):
		hawkEye.load_state_dict(torch.load(cnf.model_file,
			map_location=lambda storage, loc: storage))

	for epoch in range(cnf.epochs):
		train(epoch)
		validation(epoch)

		if (epoch+1)%10 == 0:
			torch.save(hawkEye.state_dict(), cnf.model_file)

	# end = current_milli_time()
	# print('time taken:', (end-start)*1000/60)