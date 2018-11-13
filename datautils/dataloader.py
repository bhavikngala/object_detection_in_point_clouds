import torch
from torch import from_numpy as fnp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import random
import numpy as np

from datautils.utils import *
import datautils.kittiUtils as ku
import config as cnf


class LidarLoader_2(Dataset):
	'''
	Only load train set(training is split into train and val set)
	No augmentation is done, direct training on the train data
	This model might overfit but we get a good point to start at
	'''
	def __init__(self, directory, objtype, args, train=True, augData=True):
		# load train dataset or test dataset
		self.train = train
		self.directory = directory
		self.objtype = objtype
		self.augData = args.aug_data and augData
		self.augScheme = args.aug_scheme

		# read all the filenames in the directory
		self.filenames = [join(directory, f) for f in listdir(directory) \
						  if isfile(join(directory, f))]

		# shuffle filenames
		self.filenames = random.sample(self.filenames,
			len(self.filenames))

	def __getitem__(self, index):
		filename = self.filenames[index]

		labelfilename = filename.split('/')[-1][:-4]

		# read bev
		lidarData = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

		labels = []
		# read training labels
		if self.train:
			# complete path of the label filename
			labelFilename = filename[:-10] + 'labels/' + \
							filename[-10:-4] + '.txt'

			# read lines and append them to list
			with open(labelFilename) as f:
				for line in f.readlines():
					datalist = []
					data = line.lower().strip().split()

					# object type
					if data[0] == self.objtype:
						datalist.append(1)
					# elif data[0] != 'dontcare':
						# datalist.append(0)
					else:
						continue
						# datalist.append(0)

					# convert string to float
					data = [float(data[i]) for i in range(1, len(data))]

					# TODO: is w, and l log(w) and log(l)?
					# [x, y, z, h, w, l, r]
					datalist.extend(
						[data[10], data[11], data[12], \
						 data[7], data[8], data[9], \
						 data[13]])

					labels.append(datalist)
			labels = np.zeros((1, 8), dtype=np.float32) if len(labels)==0 else np.array(labels, dtype=np.float32)	

		# augment data
		if self.train:
			if self.augData and self.augScheme == 'pixor':
				lidarData, labels[:,1:] = ku.pixorAugScheme(lidarData, labels[:,1:], self.augData)
			elif self.augData and self.augScheme == 'voxelnet':
				lidarData, labels[:,1:] = ku.voxelNetAugScheme(lidarData, labels[:,1:], self.augData)
			else:
				labels[:,1:] = ku.camera_to_lidar_box(labels[:,1:])

		bev = lidarToBEV(lidarData, cnf.gridConfig)
		labels1 = np.zeros((labels.shape[0], 7),dtype=np.float32)
		
		labels1[:,1], labels1[:,2] = np.cos(labels[:,7]), np.sin(labels[:,7])
		labels1[:,[0, 3, 4, 5, 6]] = labels[:,[0, 1, 2, 6, 5]] #class, x,y,l,w

		if labels1.shape[0] == 1 and labels1[0,0] == 0:
			z03, z12 = np.zeros((1, 6), dtype=np.float32), np.zeros((1, 6), dtype=np.float32)
			pass
		else:
			z03, z12 = self.getZoomedBoxes(labels)
			labels1[:,1:] = labels1[:, 1:] - cnf.carMean
			labels1[:,1:] = labels1[:, 1:]/cnf.carSTD

		return fnp(bev), fnp(labels1), labelfilename, z03, z12

	def __len__(self):
		return len(self.filenames)

	def getZoomedBoxes(self, labels):
		'''
		returns corners of the zoomed rectangles
		'''
		# labels: class, x, y, z, h, w, l, r
		l1 = np.copy(labels)
		l2 = np.copy(labels)
		l1[:, [5, 6]] = l1[:, [5, 6]]*0.3
		l2[:, [5, 6]] = l2[:, [5, 6]]*1.2

		z03 = ku.center_to_corner_box2d(l1[:,[1,2,5,6,7]])
		z12 = ku.center_to_corner_box2d(l2[:,[1,2,5,6,7]])

		# standarize
		z03[:, [0, 2, 4, 6]] = (z03[:, [0, 2, 4, 6]] - cnf.carMean[:,2])/cnf.carSTD[:,2]
		z03[:, [1, 3, 5, 7]] = (z03[:, [1, 3, 5, 7]] - cnf.carMean[:,3])/cnf.carSTD[:,3]
		z12[:, [0, 2, 4, 6]] = (z12[:, [0, 2, 4, 6]] - cnf.carMean[:,2])/cnf.carSTD[:,2]
		z12[:, [1, 3, 5, 7]] = (z12[:, [1, 3, 5, 7]] - cnf.carMean[:,3])/cnf.carSTD[:,3]

		return z03.reshape(labels.shape[0], -1), z12.reshape(labels.shape[0], -1) 


def collate_fn_2(batch):
	bev, labels, filenames, z03, z12 = zip(*batch)
	batchSize = len(filenames)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	# zero pad labels, zoom0_3, and zoom1_2 to make a tensor
	l = [labels[i].size(0) for i in range(batchSize)]
	m = max(l)

	labels1 = torch.zeros((batchSize, m, labels[0].size(1)))
	z03_1 = torch.zero((batchSize, m, z03[0].size(1)))
	z12_1 = torch.zero((batchSize, m, z12[0].size(1)))

	for i in range(batchSize):
		r = labels[i].size(0)
		labels1[i, :r, :] = labels[i]
		z03_1[i, :r, :] = z03[i]
		z12_1[i, :r, :] = z12[i]

	return bev, labels1, filenames, z03_1, z12_1