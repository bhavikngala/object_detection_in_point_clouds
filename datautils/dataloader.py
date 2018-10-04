import torch
from torch import from_numpy as fnp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import random
import numpy as np

from datautils.utils import lidarToBEV

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
		batch_size = batchSize, shuffle=True,
		collate_fn=collate_fn
	)
	validationLoader = DataLoader(
		LidarLoader(join(rootDir, 'val'), gridConfig, objtype),
		batch_size = batchSize, shuffle=True,
		collate_fn=collate_fn
	)
	testLoader = DataLoader(
		LidarLoader(join(rootDir, 'test'), gridConfig, train=False),
		batch_size = batchSize, shuffle=True,
		collate_fn=collate_fn
	)
	return trainLoader, validationLoader, testLoader

def collate_fn(batch):
	bev, labels, filenames = zip(*batch)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	return bev, labels, filenames

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
		filename = self.filenames[index]
		# initialize labels to a list
		labels = []

		# read binary file
		lidarData = np.fromfile(filename,
			dtype=np.float32).reshape(-1, 4)

		# convert to BEV
		bev = fnp(lidarToBEV(lidarData, self.gridConfig))

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
					# [cos(O), sin(O), dx, dy, w, l]
					datalist.extend(
						[np.cos(data[3]), np.sin(data[3]), \
						data[11], data[12], \
						data[9], data[10]])

					labels.append(datalist)
					line = f.readline()

			labels = fnp(np.array(labels, dtype='float32'))

		return bev, labels, filename

	def __len__(self):
		return len(self.filenames)