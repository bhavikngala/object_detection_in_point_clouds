import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import cv2

from networks.networks import PointCloudDetector2
from datautils.dataloader_v3 import *
from datautils.kitti_utils import KittiReader, ProjectKittiToDifferentCoordinateSystems
from datautils.kitti_utils import *
import datautils.utils as utils
import config as cnf
from lossUtils import *
import misc

import config as cnf


def saveOutput(predL, predC, projectionObject, filename, img=None, plot_img=False):
	with open('./output/v2001/labels/'+filename+'.txt', 'w') as f:
		for i in range(predL.shape[0]):
			theta = predL[i,0]
			cx = predL[i,2]
			cy = predL[i,3]
			L = predL[i,4]
			W = predL[i,5]

			l = []
			l.append('Car') # object type
			l.append(-1)    # truncation
			l.append(-1)    # occlusion
			l.append(-10)   # alpha

			# convert velo center to box corners
			temp = np.array([[cx, cy, 0, 1.5, W, L, theta]], dtype=np.float32)
			boxCorners = utils.center2BoxCorners(
				temp)
			# 2D bounding box
			imagePoints = projectionObject.project_velo_to_image(boxCorners[0])
			xmin, ymin = imagePoints.min(axis=0) # left, top
			xmax, ymax = imagePoints.max(axis=0) # right, bottom

			if (xmin < 0) or (xmax > cnf.imgWidth) or (ymin < 0) or (ymax > cnf.imgHeight):
				continue
			if (ymax-ymin)>25:
				continue

			if img is not None and plot_img:
				img = draw_projected_box3d(img, imagePoints, color=(255,0,0), thickness=2)
				cv2.imwrite('./output/v2001/img_plot/'+filename+'.png', img)

			l.append(xmin) # left
			l.append(ymin) # top
			l.append(xmax) # right
			l.append(ymax) # bottom

			# 3D dimensions
			l.append(-1)         # H
			l.append(predL[i,5]) # W
			l.append(predL[i,4]) # L

			# 3D bounding box
			rectCam = \
				projectionObject.project_velo_to_rect(
					np.array([[cx, cy, 0]]))
			l.append(rectCam[0,0]) # X
			# l.append(rectCam[0,1]) # Y
			l.append(-1000) # Y
			l.append(rectCam[0,2]) # Z

			l.append(predL[i,0]) # ry

			# set score
			l.append(predC[i])

			l = [str(t) for t in l]
			f.write(' '.join(l)+'\n')


def main():
	args = misc.getArgumentParser()

	# data loaders
	valLoader = DataLoader(
		KittiDataset(cnf, args, 'train'),
		batch_size = cnf.batchSize, shuffle=True, num_workers=3,
		collate_fn=customCollateFunction, pin_memory=True
	)

	dirList = [None, None, cnf.calTrain, cnf.leftColorTrain]
	print(dirList)
	kittiReaderObject = KittiReader(dirList)

	projectionObject = ProjectKittiToDifferentCoordinateSystems()

	targetParamObject = utils.TargetParameterization(
			gridConfig=cnf.gridConfig,
			gridL=cnf.lgrid,
			gridW=cnf.wgrid,
			downSamplingFactor=cnf.downsamplingFactor,
			device=cnf.device)


	model = PointCloudDetector2(
		cnf.res_block_layers,
		cnf.up_sample_layers,
		cnf.deconv)
	model = nn.DataParallel(model)

	model.load_state_dict(torch.load(args.model_file,
			map_location=lambda storage, loc: storage))

	model = model.to(cnf.device)

	model.eval()

	mean = cnf.carPIXORIgnoreBoundaryMean.to(cnf.device)
	std = cnf.carPIXORIgnoreBoundarySTD.to(cnf.device)

	for batchId, data in enumerate(valLoader):
		lidar, targetClasses, targetLocs, filenames = data

		lidar = lidar.cuda(cnf.device, non_blocking=True)

		classes, locs = model(lidar)
		locs = locs.permute(0, 2, 3, 1).contiguous()
		classes = classes.permute(0, 2, 3, 1).contiguous()

		for i in range(classes.size(0)):
			predC = classes[i]
			predL = locs[i]

			predC = predC.view(-1, 1)

			# filter confidence above 0.7
			mask = predC>0.7
			if mask.sum() > 0:
				# decode network output
				predL = \
					targetParamObject.decodePIXORToLabel(
						predL,
						mean,
						std)

				predL = predL.view(-1, 6)
				predC = predC[mask].cpu().detach().numpy()
				predL = predL[mask.squeeze()].cpu().detach().numpy()

				# NMS
				predL, predC = utils.nmsPredictions(predL, predC, cnf.iouThreshold)

				# read calib dict
				calibDict = kittiReaderObject.readCalibrationDict(filenames[i])
				img = kittiReaderObject.readLeftColorImage(filenames[i])
				projectionObject.clearCalibrationMatrices()
				projectionObject.setCalibrationMatrices(calibDict)

				print('img', img is None)
				print(args.plot_img)
				# save output
				saveOutput(predL, predC, projectionObject, filenames[i], img, args.plot_img)

if __name__ == '__main__':
	main()