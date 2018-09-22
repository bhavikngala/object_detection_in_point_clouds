import torch
import torch.nn.functional as F
import torch.optim as optim

from networks.networks import PointCloudDetector as HawkEye
from datautils.dataloader import *
from config import *

torch.manual_seed(0)
# select gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loaders
train_loader, vali_loader, test_loader = \
	lidarDatasetLoader(rootDir, batchSize, gridConfig, objtype)

# detector object
hawkEye = HawkEye(res_block_layers, up_sample_layers).to(device)

# network optimization method
optimizer = optim.Adam(hawkEye.parameters(), lr=learningRate)

def train(epoch):
	hawkEye.train()

	for batchId, (data, target) in enumerate(train_loader):
		# move data to GPU
		data = data.to(device)
		target = target.to(device)

		# empty the gradient buffer
		optimizer.zero_grad()

		# pass data through network and predict
		cla, loc = hawkEye(data)

		# compute loss, gradient, and optimize
		trainLoss = lossFunction(cla, loc, target)
		trainLoss.backward()
		optimizer.step()

		# TODO: compute accuracy, TP, FP, TN, FN, precision, recall

def validation(epoch):
	# TODO: validation set

if __name__ == '__main__':
	for i in range(epochs):
		train(epoch)
		validation(epoch)