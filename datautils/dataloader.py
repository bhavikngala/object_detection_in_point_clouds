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
		self.ignorebp = args.ignorebp

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
				# labels = np.ones((1, 8), dtype=np.float32)*-1
				labels = np.array([-1.0], dtype=np.float32)
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
			# z03 = np.ones((1, 8), dtype=np.float32)*-1
			# z12 = np.ones((1, 8), dtype=np.float32)*-1
			# labels1 = np.ones((1, 7), dtype=np.float32)*-1
			labels1 = np.array([-1.0], dtype=np.float32)
			targetCla = np.zeros((cnf.r, cnf.c), dtype=np.floa)
		else:
			# z03, z12 = self.getZoomedBoxes(labels)

			targetCla, targetLoc = self.encodeBoundingBoxes(labels)

		return fnp(bev), fnp(targetCla), fnp(targetLoc), labelfilename #, fnp(z03), fnp(z12)

	def __len__(self):
		return len(self.filenames)

	def getZoomedBoxes(self, labels):
		'''
		returns corners of the zoomed rectangles
		'''
		# labels: class, x, y, z, h, w, l, r
		if self.ignorebp:
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
		else:
			z1 = ku.center_to_corner_box2d(labels[:,[1,2,5,6,7]]).reshape(labels.shape[0], 8)
			z2 = z1.copy()
			# standarize
			if self.standarize:
				z1, z2 = self.normalizeZoomBoxes(z1, z2, self.norm_scheme)
			return z1, z2


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
		elif normalizeType == 'gridmean':
			labels[:,3] = (labels[:,3]-cnf.x_mean)/cnf.x_std
			labels[:,4] = (labels[:,4]-cnf.y_mean)/cnf.y_std
			labels[:,5] = labels[:,5]/cnf.lgrid
			labels[:,6] = labels[:,6]/cnf.wgrid
		else:
			labels[:, [5, 6]] = np.log(labels[:, [5, 6]])
			labels[:,1:] = labels[:,1:]-cnf.carMean
			labels[:,1:] = labels[:,1:]/cnf.carSTD
		return labels

	def normalizeZoomBoxes(self, z03, z12, normalizeType=None):
		if normalizeType=='rg':
			z03[:,[0,2,4,6]] = ((z03[:,[0,2,4,6]]-cnf.x_min)/(cnf.x_max-cnf.x_min))*(cnf.d_x_max-cnf.d_x_min)+cnf.d_x_min
			z03[:,[1,3,5,7]] = ((z03[:,[1,3,5,7]]-cnf.y_min)/(cnf.y_max-cnf.y_min))*(cnf.d_y_max-cnf.d_y_min)+cnf.d_y_min
			z12[:,[0,2,4,6]] = ((z12[:,[0,2,4,6]]-cnf.x_min)/(cnf.x_max-cnf.x_min))*(cnf.d_x_max-cnf.d_x_min)+cnf.d_x_min
			z12[:,[1,3,5,7]] = ((z12[:,[1,3,5,7]]-cnf.y_min)/(cnf.y_max-cnf.y_min))*(cnf.d_y_max-cnf.d_y_min)+cnf.d_y_min
		elif normalizeType=='gridmean':
			z03[:,[0,2,4,6]] = (z03[:,[0,2,4,6]]-cnf.x_mean)/(cnf.x_std)
			z03[:,[1,3,5,7]] = (z03[:,[1,3,5,7]]-cnf.y_mean)/(cnf.y_std)
			z12[:,[0,2,4,6]] = (z12[:,[0,2,4,6]]-cnf.x_mean)/(cnf.x_std)
			z12[:,[1,3,5,7]] = (z12[:,[1,3,5,7]]-cnf.y_mean)/(cnf.y_std)
		else:
			z03 = (z03-cnf.zoom03Mean)/cnf.zoom03STD
			z12 = (z12-cnf.zoom12Mean)/cnf.zoom12STD
		# # 2nd method
		# zb[:,[0,2,4,6]] = ((zb[:,[0,2,4,6]]-cnf.x_mean)/cnf.x_std)
		# zb[:,[1,3,5,7]] = ((zb[:,[1,3,5,7]]-cnf.y_mean)/cnf.y_std)
		return z03, z12

	def encodeBoundingBoxes(self, labels):
		'''
		Encode bounding boxes as YOLO style offsets
		'''
		x_r, y_r, z_r = cnf.gridConfig['x'], cnf.gridConfig['y'], cnf.gridConfig['z']
		res = cnf.gridConfig['res']
		ds = cnf.downsamplingFactor
		x = np.arange(x_r[1], x_r[0], -res*ds, dtype=np.float32)
		y = np.arange(y_r[0],y_r[1], res*ds, dtype=np.float32)
		xx, yy = np.meshgrid(x, y)

		r = int((y_r[1]-y_r[0])/(res*ds))
		c = int((x_r[1]-x_r[0])/(res*ds))
		targetCla = np.zeros((r, c), dtype=np.float32)
		targetLoc = np.zeros((r, c, 6), dtype=np.float32)

		# r1, c1 = labels.shape
		# labels = labels.repeat(r*c, axis=0).repeat(r1, r, c, c1)

		for i in range(labels.shape[0]):
			cl, cx, cy, cz, h, w, l, r = labels[i]

			mask = (cx <= xx) & (cx > (xx-res*ds)) & \
		           (cy >= yy) & (cy < (yy+res*ds))

			gridX = xx[mask]
			gridY = yy[mask]

			dx = (cx-gridX)/gridX
	        dy = (cy-gridY)/gridY

	        t = np.array([np.cos(2*r), np.cos(2*r), \
	                      dx, dy, l/lgrid, w/wgrid])
			targetLoc[mask] = t

			targetCla[mask] = 1.0

		if targetCla.sum() == 0:
			targetLoc = np.array([-1.0], dtype=np.float32)

		return targetCla, targetLoc


def collate_fn_3(batch):
	bev, targetCla, targetLoc, filenames = zip(*batch)
	batchSize = len(filenames)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	return bev, targetCla, targetLoc, filenames #, z03, z12