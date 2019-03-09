import torch
import torch.nn.functional as F
from torch import from_numpy as fnp
import numpy as np

import config as cnf


'''
TOY EXAMPLE

p = torch.tensor([[4,8],[7,5],[2,2]])
pr, pc = p.size()
pr = pr/cnf.batchSize
z = torch.tensor([[1,3,3,1],[1,3,9,7],[3,5,6,4],[6,8,3,1],[5,7,9,7],[0,0,0,0],[0,0,0,0]])
zr, zc = z.size()
zr = zr/cnf.batchSize
p1 = p.repeat(1, zr).view(-1, zr, pc)
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

z3 = torch.tensor([[[1,3,3,1],[1,3,9,7],[3,5,6,4],[6,8,3,1],[5,7,9,7],[0,0,0,0],[0,0,0,0]],[[3,5,3,1],[7,9,4,2],[5,7,7,5],[1,3,8,6],[0,2,11,9],[4,6,11,9],[9,11,9,7]]])
z12 = torch.tensor([[[0,4,4,0],[0,4,10,6],[2,6,7,3],[5,9,4,0],[4,8,10,6],[0,0,0,0],[0,0,0,0]],[[2,6,4,0],[6,10,5,1],[4,8,8,4],[0,4,9,6],[-1,3,12,8],[3,7,12,8],[8,12,10,6]]])
t = torch.tensor([[[1,2,2],[1,2,8],[1,4,5],[1,7,2],[0,6,8],[0,0,0],[0,0,0]],[[0,4,2],[0,8,3],[1,6,6],[1,2,7],[0,1,10],[0,5,10],[0,10,8]]])

'''

def computeIoU(matchedBoxes, targets):
	'''
	Compute Intersection over Union for 2D boxes
	'''
	f1, l1 = matchedBoxes[:,2] + matchedBoxes[:,4]/2, matchedBoxes[:,3] + matchedBoxes[:,5]/2
	b1, r1 = matchedBoxes[:,2] - matchedBoxes[:,4]/2, matchedBoxes[:,3] - matchedBoxes[:,5]/2

	f2, l2 = targets[:,2] + targets[:,4]/2, targets[:,3] + targets[:,5]/2
	b2, r2 = targets[:,2] - targets[:,4]/2, targets[:,3] - targets[:,5]/2

	intl = torch.min(torch.stack((l1, l2)), dim=0)[0]
	intr = torch.max(torch.stack((r1, r2)), dim=0)[0]
	intf = torch.min(torch.stack((f1, f2)), dim=0)[0]
	intb = torch.max(torch.stack((b1, b2)), dim=0)[0]

	intlen = intf - intb
	intwid = intl - intr

	intersectionArea = intlen * intwid
	unionArea = matchedBoxes[:,4]*matchedBoxes[:,5] + \
				targets[:, 4]*targets[:, 5] - \
				intersectionArea

	return (intersectionArea/unionArea).mean().item()


def computeLoss7(cla, loc, targetClas, targetLocs, claLossOnly=False):
	only_pos = False

	posClaLoss = None
	negClaLoss = None
	claLoss = None
	locLoss = None
	trainLoss = None
	meanConfidence = 0
	numPosSamples = 0
	numNegSamples = 0
	overallMeanConfidence = 0

	lm, lc, lh, lw = loc.size()
	lr = lh*lw
	loc = loc.permute(0, 2, 3, 1).contiguous()
	cla = cla.permute(0, 2, 3, 1).contiguous()

	for i in range(lm):
		l, c = loc[i], cla[i]
		targetCla, targetLoc = targetClas[i], targetLocs[i]

		b = (targetCla == 1).squeeze()
		numTargetsInFrame = b.sum().item()
		numPosSamples += numTargetsInFrame

		#***************PS******************
		if numTargetsInFrame:
			predL, predC, targetL = l[b], c[b], targetLoc[b]

			loss, oamc = focalLoss(predC, 1, reduction='sum', alpha=cnf.alpha)
			meanConfidence += oamc.item()
			overallMeanConfidence += oamc.item()
		
			posClaLoss = loss if posClaLoss is None else posClaLoss + loss

			if not claLossOnly:
				loss = F.smooth_l1_loss(predL, targetL, reduction='sum')
				locLoss = loss if locLoss is None else locLoss + loss
							
		#***************PS******************

		#***************NS******************
		b = (targetCla == 0).squeeze()
		numNegSamples += b.sum().item()
		predC = c[b]

		loss, oamc = focalLoss(predC, 0, reduction='sum', alpha=cnf.alpha)
			
		overallMeanConfidence += oamc.item()	
		negClaLoss = loss if negClaLoss is None else negClaLoss + loss

		#***************NS******************
	
	if numPosSamples>0:
		meanConfidence /= numPosSamples
	if numPosSamples!=0 or numNegSamples!=0:
		overallMeanConfidence /=(numPosSamples+numNegSamples)

	if numPosSamples > 0 and numNegSamples > 0:
		# posClaLoss /= (numPosSamples*cnf.accumulationSteps)
		# negClaLoss /= (numNegSamples*cnf.accumulationSteps)
		claLoss = posClaLoss + negClaLoss
		# locLoss /= (numPosSamples*cnf.accumulationSteps)
		trainLoss = claLoss if claLossOnly else claLoss + locLoss
	elif numNegSamples > 0:
		# negClaLoss /= (numNegSamples*cnf.accumulationSteps)
		claLoss = negClaLoss
		trainLoss = claLoss

	# discard loss if there are no positive samples
	if only_pos and numPosSamples==0:
		claLoss = None
		trainLoss = None

	return claLoss, locLoss, trainLoss, posClaLoss, negClaLoss, meanConfidence, overallMeanConfidence, numPosSamples, numNegSamples


def focalLoss(p, t, reduction=None, alpha = None):
	if t == 1:
		pt = p
	else:
		pt = 1 - p 
	pt.clamp_(1e-7, 1)

	logpt = torch.log(pt)
	if alpha and t == 1:
		logpt = alpha * logpt
	elif alpha and t == 0:
		logpt = (1-alpha) * logpt

	if reduction == 'mean':
		return -(((1-pt)**cnf.gamma)*logpt).mean(), pt.sum()
	elif reduction == 'sum':
		return -(((1-pt)**cnf.gamma)*logpt).sum(), pt.sum()
	else:
		return -(((1-pt)**cnf.gamma)*logpt), pt.sum()


computeLoss = computeLoss7