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

'''
TOY EXAMPLE

p = torch.tensor([[4,8],[7,5],[2,2]])
pr, pc = p.size()
pr = pr/cnf.batchSize
z = torch.tensor([[1,3,3,1],[1,3,9,7],[3,5,6,4],[6,8,3,1],[5,7,9,7],[0,0,0,0],[0,0,0,0]])
zr, zc = z.size()
zr = zr/cnf.batchSize
p1 = p.repeat(1, zr)
p1 = p1.view(-1, zr, pc)
z1 = z.repeat(1, pr, 1)
z1 = z1.view(-1, zr, zc)
zeros = z1==0
zeros = (zeros.sum(dim=-1)/zeros.size(-1)).byte()
b = (p[:,:,0]<z1[:,:,0])|(p[:,:,0]>z1[:,:,1])|(p[:,:,1]>z1[:,:,2])|(p[:,:,1]<z1[:,:,3])
c=b^zeros
c = c.sum(dim=-1)
numZeros = zeros.sum(dim=1)
numPoints = zr - numZeros
p[c==NumPoints]

p = torch.tensor([[4,8],[7,5],[2,2],[2,2],[6,2],[10,8]])
p shape = (-1, 6)
z shape = (m, points, 4)

z = torch.tensor([[[1,3,3,1],[1,3,9,7],[3,5,6,4],[6,8,3,1],[5,7,9,7],[0,0,0,0],[0,0,0,0]],[[3,5,3,1],[7,9,4,2],[5,7,7,5],[1,3,8,6],[0,2,11,9],[4,6,11,9],[9,11,9,7]]])

'''

def computeLoss3_1(cla, loc, targets, zoomed0_3, zoomed1_2):
	lm, lc, lh, lw = loc.size()
	_, zr, zc = zoomed0_3.size()
	_, tr, tc = targets.size()

	# move the channel axis to the last dimension
	loc1 = loc.permute(0, 2, 3, 1)
	cla = cla.permute(0, 2, 3, 1)

	# reshape
	loc1 = loc1.contiguous().view(-1, 6)
	cla = cla.contiguous().view(-1, 1)

	loc1 = loc1.repeat(1, zr)
	loc1 = loc1.view(-1, zr, lc)

	zoomed0_3 = zoomed0_3.repeat(1, lh*lw, 1)
	zoomed0_3 = zoomed0_3.view(-1, zr, zc)	

	zoomed1_2 = zoomed1_2.repeat(1, lh*lw, 1)
	zoomed1_2 = zoomed1_2.view(-1, zr, zc)

	targets = targets.repeat(1, lh*lw, 1)
	targets = targets.view(-1, tr, tc)

	##############~POSITIVE SAMPLES~#################
	b = ((loc1[:,:,3]<zoomed0_3[:,:,0]) & (loc1[:,:,3]>zoomed0_3[:,:,1])) & ((loc1[:,:,2]>zoomed0_3[:,:,3]) & (loc1[:,:,2]<zoomed0_3[:,:,2]))
	numPosSamples = (b.sum()).item()

	if numPosSamples>0:
		pred = cla[b.sum(dim=-1).byte()]+cnf.epsilon
		claLoss = (-cnf.alpha*(1-pred).pow(cnf.gamma)*torch.log(pred)).sum()

		locLoss = F.smooth_l1_loss(loc1[b], targets[b][:,1:])
	else:
		locLoss = None
	##############~POSITIVE SAMPLES~#################

	##############~NEGATIVE SAMPLES~#################
	zeros = zoomed1_2 == 0
	zeros = (zeros.sum(dim=-1)/zeros.size(-1)).byte()

	b = (loc1[:,:,2]<zoomed1_2[:,:,3])|(loc1[:,:,2]>zoomed1_2[:,:,2])|(loc1[:,:,3]>zoomed1_2[:,:,0])|(loc1[:,:,3]<zoomed1_2[:,:,1])
	c = b^zeros
	c = c.sum(dim=-1)

	numZeros = zeros.sum(dim=1)
	numPoints = zr - numZeros

	negPred = cla[c==numPoints]+cnf.epsilon
	numNegSamples = negPred.size(0)
	
	if numPosSamples>0 and numNegSamples>0:
		claLoss += (-cnf.alpha*negPred.pow(cnf.gamma)*torch.log(1-negPred)).sum()
	elif numNegSamples>0:
		claLoss = (-cnf.alpha*negPred.pow(cnf.gamma)*torch.log(1-negPred)).sum()
	else:
		claLoss = None

	##############~NEGATIVE SAMPLES~#################
	# print('numPosSamples:', numPosSamples, 'numNegSamples:', numNegSamples)
	return claLoss, locLoss

computeLoss = computeLoss3_1