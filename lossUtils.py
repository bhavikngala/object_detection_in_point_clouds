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

	# repeat z0_3, z1_2, and targets
	zoomed0_3 = zoomed0_3.repeat(1, lh*lw, 1)
	zoomed1_2 = zoomed1_2.repeat(1, lh*lw, 1)
	targets = targets.repeat(1, lh*lw, 1)
	zoomed0_3 = zoomed0_3.view(-1, 4)
	zoomed1_2 = zoomed1_2.view(-1, 4)
	targets = targets.view(-1, 7)

	# these are positive predictions
	posPred = ((loc[:,3]<zoomed0_3[:,0]) & (loc[:,3]>zoomed0_3[:,1])) & ((loc[:,2]>zoomed0_3[:,3]) & (loc[:,2]<zoomed0_3[:,2]))
	# these should be ignored
	ignPred = ((loc[:,3]<zoomed1_2[:,0]) & (loc[:,3]>zoomed1_2[:,1])) & ((loc[:,2]>zoomed1_2[:,3]) & (loc[:,2]<zoomed1_2[:,2]))
	# these are negative predictions
	negPred = ((loc[:,3]>zoomed1_2[:,0]) | (loc[:,3]<zoomed1_2[:,1])) | ((loc[:,2]<zoomed1_2[:,3]) | (loc[:,2]>zoomed1_2[:,2]))

	numPosSamples = posPred.sum()
	numNegSamples = negPred.sum()
	numIgnSamples = ignPred.sum()
	print('numPosSamples:', numPosSamples.item(), 'numNegSamples:', numNegSamples.item(), 'numIgnSamples:', numIgnSamples.item())

	if numPosSamples > 0:
		claLoss = (-(1-cla[posPred]).pow(cnf.gamma)*torch.log(cla[posPred])).sum()
		locLoss = F.smooth_l1_loss(loc[posPred], targets[posPred][:, 1:])
		locLoss = locLoss.mean()
	else:
		locLoss = None

	if numNegSamples > 0 and numPosSamples > 0:
		claLoss += (-cla[negPred].pow(cnf.gamma)*torch.log(1-cla[negPred])).sum()
	elif numNegSamples > 0:
		claLoss = (-cla[negPred].pow(cnf.gamma)*torch.log(1-cla[negPred])).sum()

	if numPosSamples > 0 or numNegSamples > 0:
		claLoss = claLoss/((numPosSamples+numNegSamples).float())
	else:
		claLoss = None

	return claLoss, locLoss

computeLoss = computeLoss3