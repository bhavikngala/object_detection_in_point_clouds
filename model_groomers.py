import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

class ModelTrainer():
	loader = None
	epochs = None
	model = None
	optim = None
	scheduler = None
	lossFunction = None

	def train(self, device=None):
		raise NotImplementedError()

	def val(self):
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

	def loadModel(self, filename):
		self.model.load_state_dict(torch.load(cnf.model_file,
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

	def setLossFunction(self, lossFunction):
		self.lossFunction = lossFunction