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
		self.standarize = args.standarize
		self.norm_scheme = args.norm_scheme

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
		noObjectLabels = False
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

			if not labels:
				noObjectLabels = True
				labels = np.ones((1, 8), dtype=np.float32)*-1
			else:
				labels = np.array(labels, dtype=np.float32)	

		# augment data
		if self.train:
			if self.augData and self.augScheme == 'pixor':
				lidarData, labels[:,1:] = ku.pixorAugScheme(lidarData, labels[:,1:], self.augData)
			elif self.augData and self.augScheme == 'voxelnet':
				lidarData, labels[:,1:] = ku.voxelNetAugScheme(lidarData, labels[:,1:], self.augData)
			else:
				labels[:,1:] = ku.camera_to_lidar_box(labels[:,1:])

		bev = lidarToBEV(lidarData, cnf.gridConfig)

		# remove targets outside the grid
		if not noObjectLabels:
			labels, noObjectLabels = self.getPointsInsideGrid(labels)

		if noObjectLabels:
			z03 = np.ones((1, 8), dtype=np.float32)*-1
			z12 = np.ones((1, 8), dtype=np.float32)*-1
			labels1 = np.ones((1, 7), dtype=np.float32)*-1	
		else:
			z03, z12 = self.getZoomedBoxes(labels)

			labels1 = np.zeros((labels.shape[0], 7),dtype=np.float32)
		
			labels1[:,1], labels1[:,2] = np.cos(labels[:,7]), np.sin(labels[:,7])
			labels1[:,[0, 3, 4]] = labels[:,[0, 1, 2]] #class,x,y
			labels1[:, [5, 6]] = np.log(labels[:, [6, 5]]) # logl, logw

			if self.standarize:
				labels1 = normalizeLabels(labels1, self.norm_scheme)

		return fnp(bev), fnp(labels1), labelfilename, fnp(z03), fnp(z12)

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

		z03 = ku.center_to_corner_box2d(l1[:,[1,2,5,6,7]]).reshape(labels.shape[0], 8)
		z12 = ku.center_to_corner_box2d(l2[:,[1,2,5,6,7]]).reshape(labels.shape[0], 8)

		# standarize
		if self.standarize:
			z03, z12 = self.normalizeZoomBoxes(z03, z12, self.norm_scheme)

		return z03, z12

	def getPointsInsideGrid(self, labels, grid=cnf.gridConfig):
		x_r, y_r, z_r = grid['x'], grid['y'], grid['z']
		res = grid['res']

		mask = (labels[:,2]>x_r[0]) & (labels[:,2]<x_r[1]) & (labels[:,3]>y_r[0]) & (labels[:,1]<y_r[1]) & (labels[:,3]>z_r[0]) & (labels[:,3]<z_r[1])
		if mask.sum() == 0:
			return None, True
		else:
			return labels[mask], False

	def normalizeLabels(self, labels, normalizeType=None):
		if normalizeType=='rg':
			labels[:,3] = ((labels[:,3]-cnf.x_min)/(cnf.x_max-cnf.x_min))*(cnf.d_x_max-cnf.d_x_min)+cnf.d_x_min
			labels[:,4] = ((labels[:,4]-cnf.y_min)/(cnf.y_max-cnf.y_min))*(cnf.d_y_max-cnf.d_y_min)+cnf.d_y_min
			labels[:,5] = labels[:,5]/cnf.lgrid
			labels[:,6] = labels[:,6]/cnf.wgrid
		else:
			labels[:,1:] = labels[:,1:]-cnf.carMean
			labels[:,1:] = labels[:,1:]/cnf.carSTD
		return labels

	def normalizeZoomBoxes(self, z03, z12, normalizeType=None):
		if normalizeType=='rg':
			z03[:,[0,2,4,6]] = ((z03[:,[0,2,4,6]]-cnf.x_min)/(cnf.x_max-cnf.x_min))*(cnf.d_x_max-cnf.d_x_min)+cnf.d_x_min
			z03[:,[1,3,5,7]] = ((z03[:,[1,3,5,7]]-cnf.y_min)/(cnf.y_max-cnf.y_min))*(cnf.d_y_max-cnf.d_y_min)+cnf.d_y_min
			z12[:,[0,2,4,6]] = ((z12[:,[0,2,4,6]]-cnf.x_min)/(cnf.x_max-cnf.x_min))*(cnf.d_x_max-cnf.d_x_min)+cnf.d_x_min
			z12[:,[1,3,5,7]] = ((z12[:,[1,3,5,7]]-cnf.y_min)/(cnf.y_max-cnf.y_min))*(cnf.d_y_max-cnf.d_y_min)+cnf.d_y_min
		else:
			z03 = (z03-cnf.zoom03Mean)/cnf.zoom03STD
			z12 = (z12-cnf.zoom12Mean)/cnf.zoom12STD
		# # 2nd method
		# zb[:,[0,2,4,6]] = ((zb[:,[0,2,4,6]]-cnf.x_mean)/cnf.x_std)
		# zb[:,[1,3,5,7]] = ((zb[:,[1,3,5,7]]-cnf.y_mean)/cnf.y_std)
		return z03, z12

def collate_fn_2(batch):
	bev, labels, filenames, z03, z12 = zip(*batch)
	batchSize = len(filenames)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	# zero pad labels, zoom0_3, and zoom1_2 to make a tensor
	l = [labels[i].size(0) for i in range(batchSize)]
	m = max(l)

	labels1 = torch.zeros((batchSize, m, labels[0].size(1)))
	z03_1 = torch.zeros((batchSize, m, z03[0].size(1)))
	z12_1 = torch.zeros((batchSize, m, z12[0].size(1)))

	for i in range(batchSize):
		r = labels[i].size(0)
		labels1[i, :r, :] = labels[i]
		z03_1[i, :r, :] = z03[i]
		z12_1[i, :r, :] = z12[i]

	return bev, labels1, filenames, z03_1, z12_1


def collate_fn_3(batch):
	bev, labels, filenames, z03, z12 = zip(*batch)
	batchSize = len(filenames)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	return bev, labels, filenames, z03, z12