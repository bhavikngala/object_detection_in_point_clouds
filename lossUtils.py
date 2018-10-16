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
	loc1 = loc.permute(0, 2, 3, 1)
	cla1 = cla.permute(0, 2, 3, 1)

	# reshape
	loc1 = loc1.contiguous().view(-1, 6)
	cla1 = cla1.contiguous().view(-1, 1)

	zr = zoomed0_3.size(1)
	zc = zoomed0_3.size(2)

	# repeat the loc tensor
	loc1 = loc1.repeat(1, zr)
	loc1 = loc1.view(-1, 6)
	cla1 = cla1.repeat(1, zr)
	cla1 = cla1.view(-1, 1)

	# repeat z0_3, z1_2, and targets
	zoomed0_3_1 = zoomed0_3.repeat(1, lh*lw, 1)
	targets_1 = targets.repeat(1, lh*lw, 1)
	zoomed0_3_1 = zoomed0_3_1.view(-1, 4)
	targets_1 = targets_1.view(-1, 7)

	# find all the zero tensors appended at the end of zoom boxes and targets
	zeros = zoomed0_3_1==0
	notZeros = ~((zeros.sum(dim=1)/zeros.size(1)).byte())

	# remove the tensors added as padding
	loc1 = loc1[notZeros]
	cla1 = cla1[notZeros]
	zoomed0_3_1 = zoomed0_3_1[notZeros]
	targets_1 = targets_1[notZeros]
	
	# these are positive predictions
	posPred = ((loc1[:,3]<zoomed0_3_1[:,0]) & (loc1[:,3]>zoomed0_3_1[:,1])) & ((loc1[:,2]>zoomed0_3_1[:,3]) & (loc1[:,2]<zoomed0_3_1[:,2]))

	numPosSamples = (posPred.sum()).item()

	if numPosSamples > 0:
		claLoss = (-(1-cla1[posPred]).pow(cnf.gamma)*torch.log(cla1[posPred])).sum()
		locLoss = F.smooth_l1_loss(loc1[posPred], targets_1[posPred][:, 1:])
		locLoss = locLoss.mean()
	else:
		locLoss = None
	# Part 1 end ###############################################

	# move the channel axis to the last dimension
	loc1 = loc.permute(0, 2, 3, 1)
	cla1 = cla.permute(0, 2, 3, 1)

	# reshape
	loc1 = loc1.contiguous().view(-1, 6)
	cla1 = cla1.contiguous().view(-1, 1)


	loc1 = loc1.repeat(1, zr)
	loc1 = loc1.view(-1, zr, lc)

	zoomed1_2_1 = zoomed1_2.repeat(1, lh*lw, 1)
	zoomed1_2_1 = zoomed1_2_1.view(-1, zr, zc)

	zeros = zoomed1_2_1 == 0
	zeros = (zeros.sum(dim=-1)/zeros.size(-1)).byte()

	b = (loc1[:,:,2]<zoomed1_2_1[:,:,3])|(loc1[:,:,2]>zoomed1_2_1[:,:,2])|(loc1[:,:,3]>zoomed1_2_1[:,:,0])|(loc1[:,:,3]<zoomed1_2_1[:,:,1])
	c = b^zeros
	c = c.sum(dim=-1)
	numZeros = zeros.sum(dim=1)
	numPoints = zr - numZeros
	negPred = cla1[c==numPoints]
	numNegSamples = negPred.size(0)

	if numPosSamples>0:
		claLoss += (-(negPred).pow(cnf.gamma)*torch.log(1-negPred)).sum()
	else:
		claLoss = (-(negPred).pow(cnf.gamma)*torch.log(1-negPred)).sum()

	if numPosSamples > 0 or numNegSamples > 0:
		claLoss = claLoss/(numPosSamples+numNegSamples)
	else:
		claLoss = None

	print('numPosSamples:', numPosSamples, 'numNegSamples:', numNegSamples)
	return claLoss, locLoss

computeLoss = computeLoss3