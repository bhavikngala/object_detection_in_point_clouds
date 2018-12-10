import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader


import datautils.dataloader_v3 as dataloader
import config as cnf
import misc


class ModelTrainer():
	loader = None
	epochs = None

	def train(self):
		if self.loader is None:
			print('No data to train on!!!')
			quit()

		for epoch in range(self.epochs):
			for batchId, data in enumerate(self.loader):
				lidar, targetClass, targetLoc, filenames = data

	def logStatus(self):
		pass

	def setDataloader(self, dataloader):
		self.loader = dataloader

	def setEpochs(self, epochs):
		self.epochs = epochs


def main():
	# args
	args = misc.getArgumentParser()

	# data loaders
	trainLoader = DataLoader(
		dataloader.KittiDataset(cnf, args, 'train'),
		batch_size = cnf.batchSize, shuffle=True, num_workers=3,
		collate_fn=collate_fn_3, pin_memory=True
	)

	modelTrainer = ModelTrainer()
	modelTrainer.setDataloader(trainLoader)
	modelTrainer.setEpochs(1)
	modelTrainer.train()


if __name__ == '__main__':
	main()