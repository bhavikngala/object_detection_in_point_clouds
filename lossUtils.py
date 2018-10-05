import torch
import torch.nn.functional as F
from torch import from_numpy as fnp
import numpy as np

import config as cnf

def focalLoss(pred, target, gamma=cnf.gamma):
	'''
	Focal loss function.
	ref: https://arxiv.org/pdf/1708.02002.pdf
	FL = -y*(1-p)^gamma*log(p) - (1-y)*p^gamma*lop(1-p)
	'''
	return -target*(1-pred)**gamma*torch.log(pred) - \
		   (1-target)*pred**gamma*torch.log(1-pred)

def smoothL1(loc, target):
	'''
	returns smooth l1 loss between loc and target
	'''
	return F.smooth_l1_loss(loc, target)

def computeZoomedBox(targets, zoomFactor):
	'''
	zooms the target box by zoomFactor
	'''
	zoomedTargets = []

	for target in targets:
		zoomedBoxes = []
		for i in range(target.size(0)):
			zoomedBox = [None]*4
			# left: y - w/2
			zoomedBox[0] = (target[i, 4] - target[i, 5]/2)*zoomFactor
			# rightL y + w/2
			zoomedBox[1] = (target[i, 4] + target[i, 5]/2)*zoomFactor
			# forward: x + l/2
			zoomedBox[2] = (target[i, 3] + target[i, 6]/2)*zoomFactor
			# backward: x - l/2
			zoomedBox[3] = (target[i, 3] - target[i, 6]/2)*zoomFactor

			zoomedBoxes.append(zoomedBox)
		
		zoomedTargets.append(fnp(np.array(zoomedBoxes, dtype='float32')).to(cnf.device))

	return zoomedTargets

def computeLoss(cla, loc, targets):
	'''
	Function computes classification and regression loss
	Requires: classification result of network
	Requires: localization result of network
	Requires: actual target needed to be predicted
	Requires: hyperparameter required for focal loss
	returns : total loss, L = FocalLoss + smooth_L1
	'''
	posLabel = torch.Tensor([1.0]).to(cnf.device)
	negLabel = torch.Tensor([0.0]).to(cnf.device)

	# zoom in and zoom out the target bounding boxes in the training labels
	zoomed1_2 = computeZoomedBox(targets, 1.2)
	zoomed0_3 = computeZoomedBox(targets, 0.3)

	claSamples = 0
	locSamples = 0
	claLoss = torch.Tensor([0.0]).to(cnf.device)
	locLoss = torch.Tensor([0.0]).to(cnf.device)

	for i in range(cla.size(0)):
		# reshape the output tensors for each training sample 
		# 200x175x1 -> -1x1 and 200x1175x6 -> -1x6
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)

		# get training labels and zoomed boxes for each sample
		frameTargets = targets[i]
		z1_2 = zoomed1_2[i]
		z0_3 = zoomed0_3[i]

		# for each box in predicted for a sample
		# determine whether it is positive or negative sample
		# and compute loss accordingly
		for j in range(frameLoc.size(0)):
			# predicted bounding box
			# cx, cy
			cx, cy = frameLoc[j, 2], frameLoc[j, 3]

			posSample = False
			index = -1

			# if the predicted centre falls inside any of the 0.3 zoomed bounding box
			# then it should be considered as positive sample
			for k  in range(z0_3.size(0)):
				if cy<z0_3[k][0] and cy>z0_3[k][1] and cx>z0_3[k][3] and cx<z0_3[k][2]:
					posSample = True
					index = k
					break

			if posSample:
				claLoss += focalLoss(frameCla[j], posLabel)
				locLoss += smoothL1(frameLoc[j], frameTargets[k])

				claSamples += 1
				locSamples += 1
			else:
				# if the predicted centre falls inside any of the 1.2 zoomed bounding box
				# then it should be ignored else it should be considered as negative sample
				ignoreFlag = False
				for k  in range(z1_2.size(0)):
					if cy<z1_2[k][0] and cy>z1_2[k][1] and cx>z1_2[k][3] and cx<z1_2[k][2]:
						ignoreFlag = True
						break
				if not ignoreFlag:
					claLoss += focalLoss(frameCla[j], negLabel)
					claSamples += 1

	if locSamples != 0:
		locLoss /= locSamples
	if claSamples != 0:
		claLoss /= claSamples

	return claLoss, locLoss