import torch
import torch.nn.functional as F
import torch.from_numpy as fnp
import numpy as np

import cnf as cnf

def focalLoss(pred, target, gamma=cnf.gamma):
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
			# left: y - w/2
			zoomedBox[0] = (target[i, 4] - target[i, 5]/2)*zoomFactor
			# rightL y + w/2
			zoomedBox[1] = (target[i, 4] + target[i, 5]/2)*zoomFactor
			# forward: x + l/2
			zoomedBox[2] = (target[i, 3] + target[i, 6]/2)*zoomFactor
			# backward: x - l/2
			zoomedBox[3] = (target[i, 3] - target[i, 6]/2)*zoomFactor

			zoomedBoxes.append(zoomedBox)
		
		zoomedTargets.append(fnp(np.array(zoomedBoxes, astype='float32')).to(device))

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

	zoomed1_2 = computeZoomedBox(targets, 1.2)
	zoomed0_3 = computeZoomedBox(targets, 0.3)

	claSamples = 0
	locSamples = 0
	claLoss = torch.Tensor([0.0]).to(device)
	locLoss = torch.Tensor([0.0]).to(device)

	# the points that are inside z0.3 or outside z1.2
	indices = []
	for i in range(cla.size(0)):
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)
		frameTargets = targets[i]
		z1_2 = zoomed1_2[i]
		z0_3 = zoomed0_3[i]

		for j in range(frameLoc.size(0)):
			# predicted bounding box
			# cx, cy
			cx, cy = frameLoc[j, 2], frameLoc[j, 3]

			considerFlag = False
			for k in range(z0_3.size(0)):
				if not (((cy<z1_2[k][0] and cy>z0_3[k][0]) or (cy>z1_2[k][1] and cy<z0_3[k][1])) \
					and ((cx<z1_2[k][3] and cx>z0_3[k][3]) or (cx>z1_2[k][2] and cx<z0_3[k][2]))):
					considerFlag = True
					break

			if considerFlag:
				posSample = False
				index = -1

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
					claLoss += focalLoss(frameCla[j], negLabel)
					claSamples += 1

	claLoss /= claSamples
	locLoss /= locSamples

	return claLoss, locLoss

def iou(box1, box2):
	'''
	Computes intersection over union
	box1, box2: [cx, cy, w, l]
	return IoU in case of intersection else -1
	'''
	cx1, cy1, w1, l1 = (box1[0], box1[1], box1[2], box1[3])
	cx2, cy2, w2, l2 = (box2[0], box2[1], box2[2], box2[3])

	x1, y1 = (cx1+l1/2, cy1+w1/2)
	x2, y2 = (cx2+l2/2, cy2+w2/2)

	xi_1 = np.min(x1, x2)
	xi_2 = np.max(x1-l1, x2-l2)
	yi_1 = np.min(y1, y2)
	yi_2 = np.max(y1-w1, y2-w2)

	wi = xi_1 - xi_2
	hi = yi_1 - yi_2

	if wi > 0 and hi > 0:
		return np.abs(wi*hi)/(w1*h1 + w2*h2 - wi*hi)
	else:
		return -1