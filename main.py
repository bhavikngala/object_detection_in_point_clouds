import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import time
from queue import Queue
import traceback
import argparse
import cv2

from networks.networks import PointCloudDetector2 as HawkEye
from datautils.dataloader_v2 import *
import config as cnf
from lossUtils import *
import misc

import config as cnf

parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('-r', '--root-dir', required=True, help='root directory for data')
parser.add_argument('-f', '--model-file', required=True, help='please specify model file path')
parser.add_argument('--aug-data', action='store_true')
parser.add_argument('--aug_scheme', default=None)
parser.add_argument('--norm_scheme', default=None)
parser.add_argument('-s', '--standarize', action='store_true')
parser.add_argument('--ignorebp', action='store_true')
parser.add_argument('--parameterization', default=None, help='method or target parameterization')
args = parser.parse_args()
args.standarize = True

cnf.root = args.root_dir
# data loaders
train_loader = DataLoader(
	LidarLoader_2('./../data/KITTI_BEV/9010'+'/val', cnf.calTrain, cnf.objtype, args=args, train=False),
	sampler=None, batch_size = cnf.batchSize, shuffle=False,
	num_workers=3, collate_fn=collate_fn_3, pin_memory=True
)


# create detector object and intialize weights
hawkEye = HawkEye(cnf.res_block_layers, cnf.up_sample_layers, cnf.deconv).to(cnf.device)
hawkEye = nn.DataParallel(hawkEye)
hawkEye.load_state_dict(torch.load(args.model_file,
			map_location=lambda storage, loc: storage))
hawkEye.eval()


def decodeLocPredictionsToBoxes(loc):
	x_r, y_r, z_r = cnf.gridConfig['x'], cnf.gridConfig['y'], cnf.gridConfig['z']
	res = cnf.gridConfig['res']
	ds = cnf.downsamplingFactor
	x = np.arange(x_r[1], x_r[0], -res*ds, dtype=np.float32)
	y = np.arange(y_r[0]-y_r[0]+res*ds,y_r[1]-y_r[0]+res*ds, res*ds, dtype=np.float32)
	xx, yy = np.meshgrid(x, y)

	xx = torch.from_numpy(xx)
	yy = torch.from_numpy(yy)

	loc = loc*torch.from_numpy(cnf.carSTD) + torch.from_numpy(cnf.carMean)
	loc[:,:,2] = loc[:,:,2]*xx+xx
	loc[:,:,3] = loc[:,:,3]*yy+yy+y_r[0]
	loc[:,:,4] = torch.exp(loc[:,:,4]) * cnf.lgrid
	loc[:,:,5] = torch.exp(loc[:,:,5]) * cnf.wgrid

	return loc



def getBoxesFromLocOutput(output):
	boxes = np.zeros((output.shape[0], 7))
	boxes[:,[0,1,4,5]] = output[:,[2,3,5,4]] # x,y,w,l
	boxes[:,[2]] = -1.2						 # z
	boxes[:,[3]] = 1.						 # h
	boxes[:,[6]] = (np.arctan2(output[:,1], output[:,0])/2.).reshape(-1, 1) # theta
	return boxes


def readCalibFileAndMatrices(filename):
	''' Read in a calibration file and parse into a dictionary.
	Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
	'''
	data = {}
	with open(filename, 'r') as f:
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

	V2C = data['Tr_velo_to_cam'].reshape([3,4]).astype(np.float32)
	R0 = data['R0_rect'].reshape([3,3]).astype(np.float32)
	P2 = data['P2'].reshape([3,4]).astype(np.float32)

	return V2C, R0, P2


def main():
	for batchId, data in enumerate(train_loader):
		data, targetClas, targetLocs, filenames = data

		data = data.cuda(non_blocking=True)

		# pass data through network and predict
		cla, loc = hawkEye(data)	

		for j in range(cnf.batchSize):
			targetCla = targetClas[j]


			output = loc[j].permute(1, 2, 0).contiguous().cpu().detach()
			output = decodeLocPredictionsToBoxes(output)
			prob = cla[j].permute(1, 2, 0).contiguous().cpu().detach()


			# move the channel axis to the last dimension
			output = output.view(-1, 6)
			prob = prob.view(-1, 1)


			k = min(prob.size(0), 20)
			batch_score, indices = torch.topk(prob.squeeze(), k) 
			output = output[indices]
			

			if output.size(0) > 0:
				batch_cls = ['Car']*output.size(0)
				V2C, R0, P2 = readCalibFileAndMatrices('./../data_object_calib/training/calib/'+filenames[j]+'.txt')

				boxes = getBoxesFromLocOutput(output.numpy())

				img = cv2.imread('./../data/left_color_images/data_object_image_2/training/image_2/'+filenames[j]+'.png')
				img =  ku.draw_lidar_box3d_on_image(img, boxes, None, color=(0, 255, 0), V2C=V2C, R0=R0, P2=P2)
				cv2.imwrite('./output/images/'+filenames[j]+'.png', img)
				
				labelsformat = \
					ku.box3d_to_label_1(
						boxes=list(boxes),
						cls='Car',
						scores=list(batch_score.numpy()),
						coordinate='lidar',
						V2C=V2C, R0=R0, P2=P2)

				s = ''
				for lf in list(labelsformat):
						s = s + ' '.join(lf) + '\n'

				with open('./output/labels/'+filenames[j]+'.txt', 'w') as f:
					f.write(s)

		del cla
		del loc
		del data
		del targetClas
		del targetLocs
		


if __name__ == '__main__':
	main()