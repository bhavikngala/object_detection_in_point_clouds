import torch
from torch import from_numpy as fnp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import random
import numpy as np

from utils import lidarToBEV

def lidarDatasetLoader(rootDir, batchSize, gridConfig, objtype):
	'''
	Function to create train, validation, and test loaders
	Requires: rootDir of train, validation, and test set folders
	Requires: batch size
	Requires: BEV grid configuration
	Returns: train, validation, test Dataloader objects
	'''
	trainLoader = DataLoader(
		LidarLoader(rootDir, gridConfig, objtype),
		batch_size = batchSize, shuffle=True
	)
	validationLoader = DataLoader(
		LidarLoader(rootDir+'/validation', gridConfig, objtype),
		batch_size = batchSize, shuffle=True
	)
	testLoader = DataLoader(
		LidarLoader(rootDir+'/test', gridConfig, train=False),
		batch_size = batchSize, shuffle=True
	)
	return trainLoader, validationLoader, testLoader

class LidarLoader(Dataset):
	'''
	Dataset class for LIDAR data.
	Requires directory where the LIDAR files are stored
	Requires grid configuration for lidar to BEV conversion
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
							filename[i:]

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

			labels = fnp(np.array(labels, astype='float32'))

		return bev, labels, filename

	def __len__(self):
		return len(self.filenames)