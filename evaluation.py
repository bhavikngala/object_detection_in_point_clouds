import torch
import torch.nn.functional as F
import torch.from_numpy as fnp
import numpy as np

import cnf as cnf

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

def computeTruePosFalsePosFalseNeg(cla, loc, groundTruth, iout=0.5):
	'''
	Compute the number of true positives, false positives, and
	false negatives at a given iou
	'''
	truePositives = 0
	trueNegatives = 0
	falsePositives = 0
	falseNegatives = 0

	# for each box in prediction, compute its IoU with each
	# box in groudtruth
	for i in range(cla.size(0)):
		for j in range(groundTruth.size(0)):
			piou = iou(loc[i, 2:], groundTruth[j, 2:])
			
			# if the IoU is greater than threshold
			if piou > iout:
				# true positive
				if cla[i, 0] > 0.5:
					truePositives += 1
				# false negative
				else:
					falseNegatives += 1
			else:
				# false positive
				if cla[i, 0] > 0.5:
					falsePositives += 1
				# true negative
				else:
					trueNegatives += 1

	return {
			'tp':truePositives,
			'tn':trueNegatives,
			'fp':falsePositives,
			'fn':falseNegatives
		}