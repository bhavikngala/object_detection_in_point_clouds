import torch
import torch.nn.functional as F
from torch import from_numpy as fnp
import numpy as np

import config as cnf


def computeLoss7(cla, loc, targetClas, targetLocs, alpha=1.0, beta=0.1, claLossOnly=False):
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
				loss = F.smooth_l1_loss(predC * predL, targetL, reduction='sum')
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
		posClaLoss /= numPosSamples
		negClaLoss /= numNegSamples
		claLoss = posClaLoss + negClaLoss
		locLoss /= numPosSamples
		trainLoss = alpha*claLoss if claLossOnly else alpha*claLoss + beta*locLoss
	elif numNegSamples > 0:
		negClaLoss /= numNegSamples
		claLoss = negClaLoss
		trainLoss = alpha*claLoss

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