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
	return -target*(1-pred).pow(gamma)*torch.log(pred) - \
		   (1-target)*pred.pow(gamma)*torch.log(1-pred)

def smoothL1(loc, target):
	'''
	returns smooth l1 loss between loc and target
	'''
	return F.smooth_l1_loss(loc, target)

def computeLoss(cla, loc, targets, zoomed0_3, zoomed1_2):
	'''
	Function computes classification and regression loss
	Requires: classification result of network
	Requires: localization result of network
	Requires: actual target needed to be predicted
	Requires: hyperparameter required for focal loss
	returns : total loss, L = FocalLoss + smooth_L1
	'''
	locLoss = torch.tensor([0.0], device=cnf.device)
	locSamples = 0
	claLoss = torch.tensor([0.0], device=cnf.device)
	claSamples  = 0

	for i in range(cnf.batchSize):
		# reshape the output tensors for each training sample 
		# 200x175x1 -> -1x1 and 200x175x6 -> -1x6
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)

		# for each box in predicted for a sample
		# determine whether it is positive or negative sample
		# and compute loss accordingly
		for j in range(frameLoc.size(0)):
			# predicted bounding box
			# cx, cy
			cx, cy = frameLoc[j, 2], frameLoc[j, 3]

			# if the predicted centre falls inside any of the 0.3 zoomed bounding box
			# then it should be considered as positive sample
			c = ((cy<zoomed0_3[i][:,0]) & (cy>zoomed0_3[i][:,1])) & ((cx>zoomed0_3[i][:,3]) & (cx<zoomed0_3[i][:,2]))
			matchedBox = targets[i][c].squeeze()
			
			if matchedBox.size(0) != 0:
				# focal loss
				claLoss += -(1-frameCla[j]).pow(cnf.gamma)*torch.log(frameCla[j])
				claSamples += 1
				# smooth l1 loss
				locLoss += F.smooth_l1_loss(frameLoc[j], matchedBox[1:])
				locSamples += 1
				continue

			# if the predicted center is not inside any of the zoom1_2 box
			# then it is a negative sample else ignore
			c = ((cy<zoomed1_2[i][:,0]) & (cy>zoomed1_2[i][:,1])) & ((cx>zoomed1_2[i][:,3]) & (cx<zoomed1_2[i][:,2]))
			matchedBox = targets[i][c].squeeze()

			if matchedBox.size(0) == 0:
				# focal loss
				claLoss += -frameCla[j].pow(cnf.gamma)*torch.log(1-frameCla[j])

	locLoss = locLoss/locSamples if locSamples != 0 else locLoss
	clasLoss = claLoss/claSamples if claSamples != 0 else claLoss

	return claLoss, locLoss