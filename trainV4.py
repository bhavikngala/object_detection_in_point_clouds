import torch
from torch.utils.data import DataLoader

import os
from tensorboardX import SummaryWriter

import datautils.dataloader_v3 as dataloader
import config as cnf
import misc
import networks.networks as networks
import model_groomers as mg
import lossUtils as lu


class CustomGroomer(mg.ModelTrainer):

	def __init__(self, logDir, modelFilename):
		self.writer = SummaryWriter()
		self.iter = 0
		self.logDir = logDir
		self.modelFilename = modelFilename

	def train(self, device):
		if self.loader is None:
			print('data loader is undefined')
			quit()

		for epoch in range(self.epochs):
			epochValues = []

			for batchId, data in enumerate(self.loader):
				lidar, targetClass, targetLoc, filenames = data

				lidar = lidar.cuda(device, non_blocking=True)
				targetClass = [c.cuda(device, non_blocking=True) for c in targetClass]
				targetLoc = [loc.cuda(device, non_blocking=True) for loc in targetLoc]

				predictedClass, predictedLoc = self.model(lidar)

				claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples \
				 = self.lossFunction(predictedClass, predictedLoc, targetClass, targetLoc)

				if trainLoss is not None:
					self.model.zero_grad()
					trainLoss.backward()
					self.optim.step()


				epochValues.append((claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples))
				self.iterLogger((claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples))

			self.epochLogger(epochValues, epoch)
			self.saveModel(self.modelFilename)

	def val(self):
		pass

	def iterLogger(self, values):
		claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples = \
			values
		self.writer.add_scalar('data/classification_loss', claLoss, self.iter)
		if posClaLoss is not None:
			self.writer.add_scalar('data/pos_classification_loss', posClaLoss, self.iter)
			self.writer.add_scalar('data/localization_loss', locLoss, self.iter)
			self.writer.add_scalar('data/mean_pos_sample_confidence', meanConfidence, self.iter)
		self.writer.add_scalar('data/neg_classification_loss', negClaLoss, self.iter)
		self.writer.add_scalar('data/train_loss', trainLoss, self.iter)
		self.writer.add_scalar('data/mean_pt', overallMeanConfidence, self.iter)
		self.writer.add_scalar('data/pos_samples', numPosSamples, self.iter)
		self.iter += 1

	def epochLogger(self, epochValues, epoch):
		pass'''
		pC, nC, locL, mC, mPT, nP, nN = 0, 0, 0, 0, 0, 0, 0
		for i in range(len(epochValues)):
			claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples = \
				epochValues[i]
			if posClaLoss is not None:
				pC = pC + posClaLoss*numPosSamples
				locL = locL + locLoss*numPosSamples
				mC = mC + meanConfidence*numPosSamples
			nC = nC + negClaLoss*numNegSamples
			mPT = mPT + overallMeanConfidence*(numPosSamples+numNegSamples)
			nP += numPosSamples
			nN += numNegSamples

		pC /= nP
		nC /= nN
		locL /= nP
		mC /= nP
		mPT /= (nP+nN)

		self.writer.add_scalar('train/epoch_classification_loss', pC+nC, epoch)
		self.writer.add_scalar('train/epoch_pos_classification_loss', pC, epoch)
		self.writer.add_scalar('train/epoch_neg_classification_loss', nC, epoch)
		self.writer.add_scalar('train/epoch_localization_loss', locL, epoch)
		self.writer.add_scalar('train/epoch_train_loss', pC+nC+locL, epoch)
		self.writer.add_scalar('train/epoch_mean_pos_sample_confidence', mC, epoch)
		self.writer.add_scalar('train/epoch_mean_pt', mPT, epoch)'''


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

	modelTrainer = CustomGroomer(cnf.logDir, args.model_file)
	modelTrainer.setDataloader(trainLoader)
	modelTrainer.setEpochs(1)
	modelTrainer.setModel(model)
	modelTrainer.copyModel(cnf.device)
	modelTrainer.setOptimizer('sgd', cnf.slr, cnf.momentum, cnf.decay)
	modelTrainer.setLRScheduler(cnf.lrDecay, cnf.milestones)
	

	if os.path.isfile(args.model_file):
		modelTrainer.loadModel(args.model_file)

	modelTrainer.setLossFunction(lu.computeLoss)
	modelTrainer.train(cnf.device)


if __name__ == '__main__':
	main()