import torch
import torch.nn.functional as F
import torch.from_numpy as fnp
import numpy as np

def focalLoss(gamma, pred, target):
	'''
	Focal loss function.
	ref: https://arxiv.org/pdf/1708.02002.pdf
	FL = y*(1-p)^gamma*log(p) + (1-y)*p^gamma*lop(1-p)
	'''
	return target*(1-pred)**gamma*torch.log(pred) + \
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
			# left
			zoomedBox[0] = (target[i, 4] - target[i, 5]/2)*zoomFactor
			# right
			zoomedBox[1] = (target[i, 4] + target[i, 5]/2)*zoomFactor
			# forward
			zoomedBox[2] = (target[i, 3] + target[i, 5]/2)*zoomFactor
			# backward
			zoomedBox[3] = (target[i, 3] - target[i, 5]/2)*zoomFactor

			zoomedBoxes.append(zoomedBox)
		
		zoomedTargets.append(fnp(np.array(zoomedBoxes, astype='float32')))
				

def computeLoss(cla, loc, targets, gamma):
	'''
	Function computes classification and regression loss
	Requires: classification result of network
	Requires: localization result of network
	Requires: actual target needed to be predicted
	Requires: hyperparameter required for focal loss
	returns : total loss, L = FocalLoss + smooth_L1
	'''
	zoomed1_2 = computeZoomedBox(targets, 1.2)
	zoomed0_3 = computeZoomedBox(targets, 0.3)

	