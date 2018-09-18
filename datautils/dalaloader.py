import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from os import listdir
from os.path import isfile, join
import random
import numpy as np

from utils import lidarToBEV

def lidarDatasetLoader(rootDir, batchSize,  gridConfig):
	'''
	Function to create train, validation, and test loaders
	Requires: rootDir of train, validation, and test set folders
	Requires: batch size
	Requires: BEV grid configuration
	Returns: train, validation, test Dataloader objects
	'''
	trainLoader = DataLoader(
		LidarLoader(rootDir+'/train', gridConfig),
		batch_size = batchSize, shuffle=True
	)
	validationLoader = DataLoader(
		LidarLoader(rootDir+'/validation', gridConfig),
		batch_size = batchSize, shuffle=True
	)
	testLoader = DataLoader(
		LidarLoader(rootDir+'/test', gridConfig),
		batch_size = batchSize, shuffle=True
	)
	return trainLoader, validationLoader, testLoader

class LidarLoader(Dataset):
	'''
	Dataset class for LIDAR data.
	Requires directory where the LIDAR files are stored
	Requires grid configuration for lidar to BEV conversion
	'''
	def __init__(self, directory, gridConfig):
		# read all the filenames in the directory
		self.filenames = [f for f in listdir(directory) \
						  if isfile(join(directory, f))]

		# shuffle filenames
		self.filenames = random.sample(self.filenames,
			len(self.filenames))

		# grid configuration for lidar to BEV conversion
		self.gridConfig = gridConfig

	def __getitem__(self, index):
		# TODO: read training label

		# read binary file
		lidarData = np.fromfile(self.filenames[index],
			dtype=np.float32).reshape(-1, 4)

		# convert to BEV
		bev = lidarToBEV(lidarData, self.gridConfig)

		return bev

	def __len__(self):
		return len(self.filenames)