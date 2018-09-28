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

	# according to velodyne coordinate system
	x1, y1 = (cx1+l1/2, cy1+w1/2)
	x2, y2 = (cx2+l2/2, cy2+w2/2)

	xi_1 = np.min(x1, x2)
	xi_2 = np.max(x1-l1, x2-l2)
	yi_1 = np.min(y1, y2)
	yi_2 = np.max(y1-w1, y2-w2)

	wi = max(0, xi_1 - xi_2 + 1)
	hi = max(0, yi_1 - yi_2 + 1)

	intersectionArea = wi * hi
	unionArea = w1*h1 + w2*h2 - wi*hi
	
	return intersectionArea/unionArea