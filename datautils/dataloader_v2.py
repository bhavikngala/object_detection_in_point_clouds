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

	def __init__(self, directory, calibDir, objtype, args, train=True, augData=True):
		# load train dataset or test dataset
		self.train = train
		self.directory = directory
		self.calibDir = calibDir
		self.objtype = objtype
		self.augData = args.aug_data and augData
		self.augScheme = args.aug_scheme
		self.standarize = args.standarize
		self.norm_scheme = args.norm_scheme
		self.ignorebp = args.ignorebp

		self.V2C = None
		self.C2V = None
		self.R0 = None

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

		# read training labels
		labels = []
		noObjectLabels = False
		if self.train:
			labels, noObjectLabels = self.readLabels(self.directory+'/labels/'+labelfilename+'.txt')
			calibDict = self.read_calib_file(self.calibDir+'/'+labelfilename+'.txt' )
			self.V2C = calibDict['Tr_velo_to_cam'].reshape([3,4]).astype(np.float32)
			self.R0 = calibDict['R0_rect'].reshape([3,3]).astype(np.float32)
			self.C2V = self.inverse_rigid_trans(self.V2C)
			if not noObjectLabels:
				labels[:,[1, 2, 3]] = self.project_rect_to_velo(labels[:,[1, 2, 3]]) # convert rect cam to velo cord

		# augment data
		if self.train:
			if self.augData and self.augScheme == 'pixor':
				lidarData, labels[:,1:] = ku.pixorAugScheme(lidarData, labels[:,1:], self.augData)
			elif self.augData and self.augScheme == 'voxelnet':
				lidarData, labels[:,1:] = ku.voxelNetAugScheme(lidarData, labels[:,1:], self.augData)
			# else:
			# 	labels[:,1:] = ku.camera_to_lidar_box(labels[:,1:])

		bev = lidarToBEV(lidarData, cnf.gridConfig)

		# remove targets outside the grid
		if not noObjectLabels:
			labels, noObjectLabels = self.getPointsInsideGrid(labels)

		if noObjectLabels:
			targetLoc = np.array([-1.0], dtype=np.float32)
			targetCla = np.zeros((cnf.r, cnf.c), dtype=np.float32)
		else:
			targetCla, targetLoc = self.encodeBoundingBoxes(labels)

		return fnp(bev), fnp(targetCla), fnp(targetLoc), labelfilename #, fnp(z03), fnp(z12)

	def __len__(self):
		return len(self.filenames)

	def readLabels(self, label_filename):
		# return class, x, y, z, w, h , l, r
		lines = [line.rstrip().lower().split() for line in open(label_filename)]
		lines = np.array(lines)
		lines = lines[lines[:,0]==self.objtype]
		if lines.shape[0] == 0:
			return np.zeros((1, 8), dtype=np.float32), True
		else:
			lines[:,0] = 1.0
			return lines[:,[0, 11, 12 ,13, 8, 9, 10, 14]].astype(np.float32), False

	def read_calib_file(self, filepath):
		''' Read in a calibration file and parse into a dictionary.
		Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
		'''
		data = {}
		with open(filepath, 'r') as f:
			for line in f.readlines():
				line = line.rstrip()
				if len(line)==0: continue
				key, value = line.split(':', 1)
				# The only non-float values in these files are dates, which
				# we don't care about anyway
				try:
					data[key] = np.array([float(x) for x in value.split()])
				except ValueError:
					pass

		return data

	def project_rect_to_velo(self, pts_3d_rect):
		''' Input: nx3 points in rect camera coord.
			Output: nx3 points in velodyne coord.
		''' 
		pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
		return self.project_ref_to_velo(pts_3d_ref)

	def project_rect_to_ref(self, pts_3d_rect):
		''' Input and Output are nx3 points '''
		return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

	def project_ref_to_velo(self, pts_3d_ref):
		pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
		return np.dot(pts_3d_ref, np.transpose(self.C2V))

	def cart2hom(self, pts_3d):
		''' Input: nx3 points in Cartesian
			Oupput: nx4 points in Homogeneous by pending 1
		'''
		n = pts_3d.shape[0]
		pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
		return pts_3d_hom

	def inverse_rigid_trans(self, Tr):
		''' Inverse a rigid body transform matrix (3x4 as [R|t])
			[R'|-R't; 0|1]
		'''
		inv_Tr = np.zeros_like(Tr) # 3x4
		inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
		inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
		return inv_Tr

	def getPointsInsideGrid(self, labels, grid=cnf.gridConfig):
		'''labels: Nx8 -> class, x, y, z, h, w, l, r
		'''
		x_r, y_r, z_r = grid['x'], grid['y'], grid['z']
		res = grid['res']

		mask = (labels[:,1]>x_r[0]) & (labels[:,1]<x_r[1]) & (labels[:,2]>y_r[0]) & (labels[:,2]<y_r[1]) & (labels[:,3]>z_r[0]) & (labels[:,3]<z_r[1])
		if mask.sum() == 0:
			return None, True
		else:
			return labels[mask], False

	def encodeBoundingBoxes(self, labels):
		'''
		labels: Nx8 -> class, x, y, z, w, h, l, r 
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

		numEncodedTargets = 0
		for i in range(labels.shape[0]):
			cl, cx, cy, cz, H, W, L, r = labels[i]

			mask = (cx <= xx) & (cx > (xx-res*ds)) & \
				   (cy >= yy) & (cy < (yy+res*ds))

			gridX = xx[mask]
			gridY = yy[mask]

			dx = (cx-gridX)/gridX
			dy = (cy-gridY)/gridY

			if dy>1 or dy<-1:
				continue

			l, w = np.log(L/cnf.lgrid), np.log(W/cnf.wgrid)

			t = np.array([np.cos(2*r), np.sin(2*r), \
						  dx, dy, \
						  l, w])

			if self.standarize:
				t = (t-cnf.carMean)/cnf.carSTD

			targetLoc[mask] = t

			targetCla[mask] = 1.0

			numEncodedTargets += 0

		if targetCla.sum() == 0:
			targetLoc = np.array([-1.0], dtype=np.float32)
		if numEncodedTargets == 0:
			targetCla = np.zeros((r, c), dtype=np.float32)
			targetLoc = np.array([-1.0], dtype=np.float32)

		return targetCla, targetLoc

def collate_fn_3(batch):
	bev, targetCla, targetLoc, filenames = zip(*batch)
	batchSize = len(filenames)

	# Merge bev (from tuple of 3D tensor to 4D tensor).
	bev = torch.stack(bev, 0)

	return bev, targetCla, targetLoc, filenames #, z03, z12