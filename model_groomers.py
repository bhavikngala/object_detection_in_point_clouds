import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import math

class ModelTrainer():
	loader = None
	epochs = None
	model = None
	optim = None
	scheduler = None
	lossFunction = None
	trainLoader = None
	valLoader = None
	testLoader = None
	lrRange = None
	momentumRange = None
	stepSize = None
	oneCycleLearning = False
	alpha1 = 1.0
	beta1 = 0.1


	def train(self, device=None):
		raise NotImplementedError()


	def val(self):
		raise NotImplementedError()


	def setLoaders(self):
		raise NotImplementedError()


	def setDataloader(self, dataloader):
		self.loader = dataloader


	def setEpochs(self, epochs):
		self.epochs = epochs


	def setModel(self, model):
		self.model = model


	def setOptimizer(self, optimName, lr, momentum=0, decay=0, nesterov=False, betas=(0.9, 0.999)):
		if self.model is None:
			print('model undefined')
			quit()
		if optimName == 'sgd':
			self.optim = SGD(self.model.parameters(), lr, momentum=momentum, weight_decay=decay, nesterov=nesterov)
		elif optimName == 'adam':
			self.optim = Adam(self.model.parameters(), lr, betas=betas, weight_decay=decay)


	def setLRScheduler(self, lrDecay, milestones):
		if self.optim is None:
			print('optimizer undefined')
			quit()
		self.scheduler = MultiStepLR(self.optim, milestones, lrDecay)


	def oneCycleStep(self, epoch):
		if self.lrRange is None or self.momentumRange is None:
			print('lrRange and momentumRange undefined')
			quit()

		cycle = math.floor(1 + epoch/(2*self.stepSize))
		x = math.fabs(epoch/self.stepSize - 2*cycle + 1)
		locallr = self.lrRange[0] + (self.lrRange[1]-self.lrRange[0])*max(0, 1-x)
		localMomentum = self.momentumRange[1] - (self.momentumRange[1]-self.momentumRange[0])*max(0, 1-x)
		for param_group in self.optim.param_groups:
			param_group['lr'] = locallr
			param_group['momentumRange'] = localMomentum


	def setLrRangeStepSize(self, lrRange, momentumRange, stepSize):
		self.lrRange = lrRange
		self.momentumRange = momentumRange
		self.stepSize = stepSize
		self.oneCycleLearning = True


	def getLrRangeStepSize(self):
		return self.lrRange, self.stepSize


	def loadModel(self, filename):
		self.model.load_state_dict(torch.load(filename,
			map_location=lambda storage, loc: storage))


	def saveModel(self, filename):
		if self.model is None:
			print('Model is None, cannot save')
			quit()
		torch.save(self.model.state_dict(), filename)


	def copyModel(self, device):
		if self.model is None:
			print('Model is None, cannot copy')
			quit()
		self.model.cuda(device)


	def setLossFunction(self, lossFunction, alpha1, beta1):
		self.lossFunction = lossFunction
		self.alpha1 = alpha1
		self.beta1 = beta1


	def setDataParallel(self, flag):
		if flag:
			self.model = nn.DataParallel(self.model)