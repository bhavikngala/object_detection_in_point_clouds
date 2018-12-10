import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

class ModelTrainer():
	loader = None
	epochs = None
	model = None
	optim = None
	scheduler = None

	def train(self):
		throw NotImplementedError

	def logStatus(self):
		pass

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