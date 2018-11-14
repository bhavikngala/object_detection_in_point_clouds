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

def findInOutMask(loc, rectangle, inside=True):
	# rectangle is array of 4 points of 2d bounding box
	# find vectors of 2 adjacent sides
	# AB = (Bx-Ax, By-Ay) and so on
	# point inside rectangle ABCD should satisfy the condition
	# 0 <= dot(AB, AM) <= dot(AB,AB) and
	# 0 <= dot(BC, BM) <= dot(BC,BC)
	# AB
	AB_x = rectangle[:, :, 2] - rectangle[:, :, 0] # Bx - Ax
	AB_y = rectangle[:, :, 3] - rectangle[:, :, 1] # By - Ay

	# BC
	BC_x = rectangle[:, :, 4] - rectangle[:, :, 2] # Cx - Bx
	BC_y = rectangle[:, :, 5] - rectangle[:, :, 3] # Cy - By

	# AM
	AM_x = loc[:, :, 2] - rectangle[:, :, 0] # Mx - Ax
	AM_y = loc[:, :, 3] - rectangle[:, :, 1] # My - Ay

	# BM
	BM_x = loc[:, :, 2] - rectangle[:, :, 2] # Mx - Bx
	BM_y = loc[:, :, 3] - rectangle[:, :, 3] # My - By

	dot_AB_AM = AB_x*AM_x + AB_y*AM_y
	dot_AB_AB = AB_x*AB_x + AB_y*AB_y
	dot_BC_BM = BC_x*BM_x + BC_y*BM_y
	dot_BC_BC = BC_x*BC_x + BC_y*BC_y

	if inside:
		mask = (0<dot_AB_AM) & (dot_AB_AM<dot_AB_AB) & (0<dot_BC_BM) & (dot_BC_BM<dot_BC_BC)
	else:
		mask = ~((0<dot_AB_AM) & (dot_AB_AM<dot_AB_AB) & (0<dot_BC_BM) & (dot_BC_BM<dot_BC_BC))
	
	return mask

def computeLoss4_1(cla, loc, targets, zoomed0_3, zoomed1_2):
	claLoss = None
	locLoss = None
	iou = None
	meanConfidence = None

	lm, lc, lh, lw = loc.size()
	_, zr, zc = zoomed0_3.size()
	_, tr, tc = targets.size()

	# move the channel axis to the last dimension
	loc1 = loc.permute(0, 2, 3, 1).contiguous().view(lm, lw*lh, lc)
	cla = cla.permute(0, 2, 3, 1).contiguous().view(lm, lw*lh, 1)

	loc1 = loc1.repeat(1, 1, zr).view(lm, -1, lc)
	cla1 = cla.repeat(1, 1, zr).view(lm, -1, 1)

	zoomed0_3 = zoomed0_3.repeat(1, lh*lw, 1)
	zoomed1_2 = zoomed1_2.repeat(1, lh*lw, 1)
	targets = targets.repeat(1, lh*lw, 1)

	#***************PS******************
	
	b = findInOutMask(loc1, zoomed0_3, inside=True)
	numPosSamples = b.sum().item()
	
	if numPosSamples>0:
		pt = cla1[b]
		pt.clamp_(1e-7, 1-1e-7)
		logpt = torch.log(pt)
		claLoss = cnf.alpha*(-((1-pt)**cnf.gamma)*logpt).mean()

		locLoss = F.smooth_l1_loss(loc1[b], targets[b][:,1:])

		# iou = computeIoU(loc1[b], targets[b][:,1:])
		iou = 0
		meanConfidence = pt.mean().item()

	#***************PS******************

	#***************NS******************
	
	b1 = findInOutMask(loc1, zoomed1_2, inside=False)
	b1 = b1.view(lm, lw*lh, zr).sum(dim=-1)==zr

	numNegSamples = b1.sum().item()

	if numNegSamples>0 and numPosSamples>0:
		cla1 = cla1.view(lm, lw*lh, 1*zr)

		pt = 1-cla1[b1][:,0]
		pt.clamp_(1e-7, 1-1e-7)
		logpt = torch.log(pt)
		claLoss += cnf.alpha*(-((1-pt)**cnf.gamma)*logpt).mean()
	else:
		cla1 = cla1.view(lm, lw*lh, 1*zr)

		pt = 1-cla1[b1][:,0]
		pt.clamp_(1e-7, 1-1e-7)
		logpt = torch.log(pt)
		claLoss = cnf.alpha*(-((1-pt)**cnf.gamma)*logpt).mean()

	#***************NS******************
	
	return claLoss, locLoss, iou, meanConfidence, numPosSamples, numNegSamples

computeLoss = computeLoss4_1