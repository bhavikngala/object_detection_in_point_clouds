import torch
import torch.nn.functional as F
from config import *

def focalLoss(gamma, pred, target):
	'''
	Focal loss function.
	ref: https://arxiv.org/pdf/1708.02002.pdf
	FP = y*(1-p)^gamma*log(p) + (1-y)*p^gamma*lop(1-p)
	'''
	return target*(1-pred)**gamma*torch.log(pred) + \
		   (1-target)*pred**gamma*torch.log(1-pred)