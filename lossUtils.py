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

def computeLoss1(cla, loc, targets, zoomed0_3, zoomed1_2):
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


def computeLoss2(cla, loc, targets, zoomed0_3, zoomed1_2):
	locLoss = torch.tensor([0.0], device=cnf.device)
	locSamples = 0
	claLoss = torch.tensor([0.0], device=cnf.device)
	claSamples  = 0

	for i in range(cnf.batchSize):
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)
		target = targets[i]

		z0_3 = zoomed0_3[i]
		z1_2 = zoomed1_2[i]

		zr, zc = z0_3.size()
		fr, fc = frameLoc.size()

		# repeat the loc tensor
		frameLoc = frameLoc.repeat(1, zr)
		frameLoc = frameLoc.view(-1, 6)
		frameCla = frameCla.repeat(1, zr)
		frameCla = frameCla.view(-1, 1)

		# repeat z0_3 and z1_2
		z0_3 = z0_3.repeat(fr, 1)
		z1_2 = z1_2.repeat(fr, 1)

		c = ((frameLoc[:,3]<z0_3[:,0]) & (frameLoc[:,3]>z0_3[:,1])) & ((frameLoc[:,2]>z0_3[:,3]) & (frameLoc[:,2]<z0_3[:,2]))
		matchedBoxes = frameLoc[c]

		if matchedBox.size(0) != 0:
			# focal loss
			claLoss += -(1-frameCla[c]).pow(cnf.gamma)*torch.log(frameCla[c])
			claSamples += 1
			# smooth l1 loss
			locLoss += F.smooth_l1_loss(frameLoc[j], matchedBox[1:])
			locSamples += 1

def computeLoss3(cla, loc, targets, zoomed0_3, zoomed1_2):
	lm, lc, lh, lw = loc.size()

	# move the channel axis to the last dimension
	loc = loc.permute(0, 2, 3, 1)
	cla = cla.permute(0, 2, 3, 1)

	# reshape
	loc = loc.contiguous().view(-1, 6)
	cla = cla.contiguous().view(-1, 6)

	zr = zoomed0_3.size(1)

	# repeat the loc tensor
	loc = loc.repeat(1, zr)
	loc = loc.view(-1, 6)
	cla = cla.repeat(1, zr)
	cla = cla.view(-1, 1)

	# repeat z0_3 and z1_2
	zoomed0_3 = zoomed0_3.repeat(1, lh*lw, 1)
	zoomed1_2 = zoomed1_2.repeat(1, lh*lw, 1)
	zoomed0_3 = zoomed0_3.view(-1, 4)
	zoomed1_2 = zoomed1_2.view(-1, 4)

	# these are positive predictions
	posPred = ((loc[:,3]<zoomed0_3[:,0]) & (loc[:,3]>zoomed0_3[:,1])) & ((loc[:,2]>zoomed0_3[:,3]) & (loc[:,2]<zoomed0_3[:,2]))
	# these should be ignore
	ignPred = ((loc[:,3]<zoomed1_2[:,0]) & (loc[:,3]>zoomed1_2[:,1])) & ((loc[:,2]>zoomed1_2[:,3]) & (loc[:,2]<zoomed1_2[:,2]))
	# these are negative predictions
	negPred = ~ignPred

	numPosSamples = posPred.sum()
	numNegSamples = negPred.sum()
	print('numPosSamples:', numPosSamples.item(), 'numNegSamples:', numPosSamples.item())

	if numPosSamples > 0:
		claLoss = (-(1-cla[posPred]).pow(cnf.gamma)*torch.log(cla[posPred])).sum()
		locLoss = F.smooth_l1_loss(loc[posPred], targets[posPred][1:])
		locLoss = locLoss.mean()
	else:
		locLoss = None

	if numNegSamples > 0 and numPosSamples > 0:
		clasLoss += (-cla[posPred].pow(cnf.gamma)*torch.log(1-cla[posPred])).sum()
	elif numNegSamples > 0:
		claLoss = (-cla[posPred].pow(cnf.gamma)*torch.log(1-cla[posPred])).sum()

	if numPosSamples > 0 or numNegSamples > 0:
		# claLoss = claLoss/((numPosSamples+numNegSamples).float())
		claLoss = claLoss
	else:
		claLoss = None

	return claLoss, locLoss

computeLoss = computeLoss3