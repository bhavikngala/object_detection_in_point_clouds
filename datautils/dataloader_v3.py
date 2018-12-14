import torch
from torch import from_numpy as fnp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from os import listdir
from os.path import isfile, join
import random
import numpy as np

import datautils.utils as utils
from datautils.kitti_utils import *


class KittiDataset(Dataset):

	def __init__(self, cnf, args, dataSetType):
		self.directory = cnf.rootDir
		self.innerFolder = 'testing' if dataSetType=='test' else 'training'
		
		if dataSetType=='test':
			splitFile = cnf.testFile
		elif dataSetType=='train' and args.full_train_set:
			splitFile = cnf.fullTrainFile
		elif dataSetType=='train':
			splitFile  = cnf.trainSplitFile
		else:
			splitFile = cnf.valSplitFile

		with open(splitFile, 'r') as f:
			self.filenames = [line.strip() for line in f.readlines()]

		self.calibDir = cnf.calTest if dataSetType=='test' else cnf.calTrain
		self.lidarDir = cnf.rootDir
		
		self.dirList = [os.path.join(self.lidarDir,self.innerFolder, 'velodyne'),
						os.path.join(self.lidarDir,self.innerFolder,'label_2'),
						self.calibDir,
						None]

		self.objectType = cnf.objtype
		self.loadLabels = False if dataSetType=='test' else True
		self.grid = cnf.gridConfig

		self.targetParamObject = utils.TargetParameterization(
			gridConfig=cnf.gridConfig,
			gridL=cnf.lgrid,
			gridW=cnf.wgrid,
			downSamplingFactor=cnf.downsamplingFactor,
			device=None)
		self.kittiReaderObject = KittiReader(self.dirList)
		self.projectionObject = ProjectKittiToDifferentCoordinateSystems()
		self.r = cnf.r
		self.c = cnf.c
		self.mean=cnf.carPIXORIgnoreBoundaryMean,
		self.std=cnf.carPIXORIgnoreBoundarySTD


	def __len__(self):
		return len(self.filenames)


	def __getitem__(self, index):
		filename = self.filenames[index]

		lidarData = self.kittiReaderObject.readLidarFile(filename)
		calibDict = self.kittiReaderObject.readCalibrationDict(filename)
		self.projectionObject.clearCalibrationMatrices()
		self.projectionObject.setCalibrationMatrices(calibDict)
		
		if self.loadLabels:
			# labels -> c, _, _, alpha, _, _, _, _, h, w, l, x, y, z, ry, score
			labels = self.kittiReaderObject.readLabels(filename)
			# get labels required for the network
			labels = self.formatLabelsToUseCase(labels)
			if labels is not None: # if labels with required objects are present
				# project cam to velodyne
				labels[:,[11, 12, 13]] = self.projectionObject.project_rect_to_velo(labels[:,[11, 12, 13]])
				labels = self.getLabelsInsideGrid(labels)
			if labels is not None: # if required objects are inside grid
				# target parameterization
				labels = fnp(labels)
				targetClass, targetLoc = self.targetParamObject.encodeLabelToPIXORIgnoreBoundaryPix(labels[:,[0,11,12,13,8,9,10,14]], self.mean, self.std)
			else:
				targetClass = torch.zeros((self.r, self.c), dtype=torch.float32)
				targetLoc = torch.tensor([-1.], dtype=torch.float32)
		bev = utils.lidarToBEV(lidarData, self.grid)

		return fnp(bev), targetClass, targetLoc, filename


	def formatLabelsToUseCase(self, labels):
		labels = labels[labels[:,0]==self.objectType]
		if labels.shape[0] == 0:
			return None
		labels[:,0] = '1.0'
		labels = labels.astype(np.float32)
		return labels


	def getLabelsInsideGrid(self, labels):
		mask = \
			((labels[:,11]<=self.grid['x'][1]) & (labels[:,11]>=self.grid['x'][0])) & \
			((labels[:,12]<=self.grid['y'][1]) & (labels[:,12]>=self.grid['y'][0])) & \
			((labels[:,13]<=self.grid['z'][1]) & (labels[:,13]>=self.grid['z'][0]))

		if mask.sum() > 0:
			return labels[mask]
		else:
			return None


def customCollateFunction(batch):
	bev, targetCla, targetLoc, filenames = zip(*batch)
	bev = torch.stack(bev, 0)
	return bev, targetCla, targetLoc, filenames