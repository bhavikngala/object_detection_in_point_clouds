import torch
from torch.utils.data import DataLoader


import datautils.dataloader_v3 as dataloader
import config as cnf
import misc
import networks.networks as networks
import model_groomers as mg


class CustomGroomer(mg.ModelTrainer):

	def train(self):
		if self.loader is None:
			print('No data to train on!!!')
			quit()

		n = 0
		for epoch in range(self.epochs):
			for batchId, data in enumerate(self.loader):
				lidar, targetClass, targetLoc, filenames = data
				n += len(filenames)
				print(filenames)

		print('num samples:', n)


def main():
	# args
	args = misc.getArgumentParser()

	# data loaders
	trainLoader = DataLoader(
		dataloader.KittiDataset(cnf, args, 'train'),
		batch_size = cnf.batchSize, shuffle=True, num_workers=3,
		collate_fn=dataloader.customCollateFunction, pin_memory=True
	)

	# define model
	model = networks.PointCloudDetector2(
		cnf.res_block_layers,
		cnf.up_sample_layers,
		cnf.deconv)

	modelTrainer = CustomGroomer()
	modelTrainer.setDataloader(trainLoader)
	modelTrainer.setEpochs(cnf.epochs)
	modelTrainer.setModel(model)
	modelTrainer.setOptimizer('sgd', cnf.slr, cnf.momentum, cnf.decay)
	modelTrainer.setLRScheduler(cnf.lrDecay, cnf.milestones)
	modelTrainer.train()


if __name__ == '__main__':
	main()