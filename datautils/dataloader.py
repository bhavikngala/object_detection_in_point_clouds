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
	bev, labels, filenames, zoom0_3, zoom1_2 = zip(*batch)
	batchSize = len(zoom1_2)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	# zero pad labels, zoom0_3, and zoom1_2 to make a tensor
	l = [len(labels[i]) for i in range(batchSize)]
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
		lidar, labels, filename = self.readData(index)

		# transform if it is train, val set
		if self.train:
			# randomly select to augment a frame or not
			transformation = np.random.randint(low=0, high=4)

			if transformation == 0:
				# rotate the frame
				lidar, labels = rotateFrame(lidar, labels)
			elif transformation == 1:
				# scale frame
				lidar, labels = scaleFrame(lidar, labels)
			elif transformation == 2:
				# perturb frame
				lidar, labels = perturbFrame(lidar, labels)
			# transformation == 3, no augmentation is applied

		# convert lidar to BEV
		bev = fnp(lidarToBEV(lidar.numpy(), self.gridConfig))

		# enlarge the box by a factor of 1.2 and a factor of 0.3
		zoom1_2 = torch.zeros(labels.size())
		zoom0_3 = torch.zeros(labels.size())

		# left: y - w/2, rightL y + w/2, forward: x + l/2, backward: x - l/2
		zoom1_2[:, 0] = labels[:, 4] - labels[:, 5]*0.6
		zoom1_2[:, 1] = labels[:, 4] + labels[:, 5]*0.6
		zoom1_2[:, 2] = labels[:, 3] + labels[:, 6]*0.15
		zoom1_2[:, 3] = labels[:, 3] - labels[:, 6]*0.15

		zoom0_3[:, 0] = labels[:, 4] - labels[:, 5]*0.6
		zoom0_3[:, 1] = labels[:, 4] + labels[:, 5]*0.6
		zoom0_3[:, 2] = labels[:, 3] + labels[:, 6]*0.15
		zoom0_3[:, 3] = labels[:, 3] - labels[:, 6]*0.15

		return bev, labels, filename, zoom0_3, zoom1_2

	def readData(self, index):
		filename = self.filenames[index]
		# initialize labels to a list
		labels = []
		zoom1_2 = None
		zoom0_3 = None

		# read binary file
		lidarData = fnp(np.fromfile(filename, dtype=np.float32).reshape(-1, 4))

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

		return lidarData, labels, filename

	def __len__(self):
		return len(self.filenames)