import torch
from torch import from_numpy as fnp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import random
import numpy as np

from datautils.utils import lidarToBEV, rotateFrame, scaleFrame, perturbFrame

def lidarDatasetLoader(rootDir, batchSize, gridConfig, objtype):
	'''
	Function to create train, validation, and test loaders
	Requires: rootDir of train, validation, and test set folders
	Requires: batch size
	Requires: BEV grid configuration
	Returns: train, validation, test Dataloader objects
	'''
	trainLoader = DataLoader(
		LidarLoader(join(rootDir, 'train'), gridConfig, objtype),
		batch_size = batchSize, shuffle=True, num_workers=2,
		collate_fn=collate_fn, pin_memory=True
	)
	validationLoader = DataLoader(
		LidarLoader(join(rootDir, 'val'), gridConfig, objtype),
		batch_size = batchSize, shuffle=True, num_workers=2,
		collate_fn=collate_fn, pin_memory=True
	)
	testLoader = DataLoader(
		LidarLoader(join(rootDir, 'test'), gridConfig, train=False),
		batch_size = batchSize, shuffle=True, num_workers=2,
		collate_fn=collate_fn, pin_memory=True
	)
	return trainLoader, validationLoader, testLoader

def collate_fn(batch):
	bev, labels, filenames, zoom0_3, zoom1_2 = zip(*batch)
	batchSize = len(zoom1_2)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	# zero pad labels, zoom0_3, and zoom1_2 to make a tensor
	l = [labels[i].size(0) for i in range(batchSize)]
	m = max(l)
	labels1 = torch.zeros((batchSize, m, labels[0].size(1)))
	zoom0_3_1 = torch.zeros((batchSize, m, zoom0_3[0].size(1)))
	zoom1_2_1 = torch.zeros((batchSize, m, zoom1_2[0].size(1)))

	for i in range(batchSize):
		r = labels[i].size(0)
		r1 = zoom0_3[i].size(0)
		labels1[i, :r, :] = labels[i]
		zoom0_3_1[i, :r1, :] = zoom0_3[i]
		zoom1_2_1[i, :r1, :] = zoom1_2[i]

	return bev, labels1, filenames, zoom0_3_1, zoom1_2_1

class LidarLoader(Dataset):
	'''
	Dataset class for LIDAR data.
	Requires directory where the LIDAR files are stored
	Requires grid configuration for lidar to BEV conversion
	Returns: data - bird's eye view of the lidar point cloud, tensor
			 labels - tuple of label tensors of each file
			 filenames - tuple of filenames of the lidar files in the batch 
	'''
	def __init__(self, directory, gridConfig, objtype=None, train=True):
		# object tyoe
		self.objtype = objtype
		# load train dataset or test dataset
		self.train = train

		# read all the filenames in the directory
		self.filenames = [join(directory, f) for f in listdir(directory) \
						  if isfile(join(directory, f))]

		# shuffle filenames
		self.filenames = random.sample(self.filenames,
			len(self.filenames))

		# grid configuration for lidar to BEV conversion
		self.gridConfig = gridConfig

	def __getitem__(self, index):
		# read data for a frame at index
		lidar, labels, filename = self.readData(index)

		# transform if it is train, val set
		if self.train:
			# randomly select to augment a frame or not
			transformation = np.random.randint(low=0, high=4)

			if transformation == 0:
				# rotate the frame
				lidar, labels[:, 1:] = rotateFrame(lidar, labels[:, 1:])
			elif transformation == 1:
				# scale frame
				lidar, labels[:, 1:] = scaleFrame(lidar, labels[:, 1:])
			elif transformation == 2:
				# perturb frame
				lidar, labels[:, 1:] = perturbFrame(lidar, labels[:, 1:])
			# transformation == 3, no augmentation is applied

		# convert lidar to BEV
		bev = fnp(lidarToBEV(lidar, self.gridConfig))
		labels = self.convertLabelsToOutputTensor(fnp(labels))
		zoom0_3, zoom1_2 = self.getZoomedBoxes(labels)

		return bev, labels, filename, zoom0_3, zoom1_2

	def readData(self, index):
		'''
		Returns: lidar - numpy array of shape(-1, 4); [x, y, z, reflectance]
		Returns: labels - numpy array of shape(-1, 8); [class, x, y, z, h, w, l, ry]
		Returns: filename - string
		'''
		filename = self.filenames[index]
		# initialize labels to a list
		labels = []
		zoom1_2 = None
		zoom0_3 = None

		# read binary file
		lidarData = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

		# read training labels
		if self.train:
			# complete path of the label filename
			i = filename.rfind('/')
			labelFilename = filename[:i] + '/labels' + \
							filename[i:-4] + '.txt'

			# read lines and append them to list
			with open(labelFilename) as f:
				line = f.readline()
				while line:
					datalist = []
					data = line.split()

					# object type
					if data[0] == self.objtype:
						datalist.append(1)
					else:
						datalist.append(0)

					# convert string to float
					data = [float(data[i]) for i in range(1, len(data))]

					# TODO: is w, and l log(w) and log(l)?
					# [x, y, z, h, w, l, r]
					datalist.extend(
						[data[10], data[11], data[12], data[7], \
						 data[8], data[9], data[13]])

					labels.append(datalist)
					line = f.readline()

		labels = np.array(labels)
		return lidarData, labels, filename

	def getZoomedBoxes(self, labels, factor1=0.3, factor2=1.2):
		factor1 = factor1/2
		factor2 = factor2/2

		zoom1_2 = torch.zeros((labels.size(0), 4))
		zoom0_3 = torch.zeros((labels.size(0), 4))

		# left: y + w/2, right: y - w/2, forward: x + l/2, backward: x - l/2
		zoom1_2[:, 0] = labels[:, 4] + labels[:, 6]*factor2
		zoom1_2[:, 1] = labels[:, 4] - labels[:, 6]*factor2
		zoom1_2[:, 2] = labels[:, 3] + labels[:, 5]*factor2
		zoom1_2[:, 3] = labels[:, 3] - labels[:, 5]*factor2

		zoom0_3[:, 0] = labels[:, 4] + labels[:, 6]*factor1
		zoom0_3[:, 1] = labels[:, 4] - labels[:, 6]*factor1
		zoom0_3[:, 2] = labels[:, 3] + labels[:, 5]*factor1
		zoom0_3[:, 3] = labels[:, 3] - labels[:, 5]*factor1

		return zoom0_3, zoom1_2

	def convertLabelsToOutputTensor(self, labels):
		'''
		Requires: labels - tensor of shape(-1, 8); [class, x, y, z, h, w, l, ry]
		Returns : tensor of shape(-1, 7); [class, cos(theta), sin(theta), x, y, l, w]
		'''
		ret = torch.zeros(labels.size(0), 7)
		
		# class
		ret[:, 0] = labels[:, 0]
		# cos(theta), sin(theta)
		ret[:, 1], ret[:, 2] = torch.cos(labels[:, 7]), torch.sin(labels[:, 7])
		# cx, cy, l, w
		ret[:, 3], ret[:, 4] = labels[:, 1], labels[:, 2]
		ret[:, 5], ret[:, 6] = labels[:, 6], labels[:, 5]

		return ret

	def __len__(self):
		return len(self.filenames)