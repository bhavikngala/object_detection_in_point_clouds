import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import os
import time
import traceback
from queue import Queue
import argparse

from networks.networks import PointCloudDetector
from networks.resnet import ResNet18
from datautils.dataloader import *
import config as cnf
from lossUtils import computeLoss
import misc

parser = argparse.ArgumentParser(description='Train network')
parser.add_argument('--step-lr', action='store_true', help='to set step learing rate or not')
parser.add_argument('--aug-data', action='store_true', help='to augment data or to not')
parser.add_argument('-f', '--model-file', default=None, help='used to set different model file name other than default one')
parser.add_argument('-r', '--root-dir', default=None, help='root directory for data')
parser.add_argument('-p', '--pixor', action='store_true', help='use pixor augmentation scheme')
parser.add_argument('-v', '--voxelnet', action='store_true', help='use voxelnet augmentation scheme')
parser.add_argument('-e', '--epochs', type=int, default=None, help='used to specify number of epochs other than the default value set in config')
parser.add_argument('--aug-scheme', default=None, help='augmentation scheme string used in DataLoader')
parser.add_argument('-m', '--multi-gpu', action='store_true', help='use multiple gpus if set')
parser.add_argument('--val', action='store_true', help='evaluate on validation set during training or not')
parser.add_argument('-c', '--clip', action='store_true', help='whether to clip gradients or not')
parser.add_argument('--clipvalue', type=float, default=0.25, help='gradient norm value threshold for clipping')
parser.add_argument('--resnet18', action='store_true', help='use standard resent 18 as backbone')
parser.add_argument('-s', '--standarize', action='store_true', help='specify whether to standarize the target or not')
parser.add_argument('-d', '--discard', action='store_true', help='used to perform negative hard mining, only consider top k negative samples')
parser.add_argument('-o', '--only-pos', action='store_true', help='specify whether to consider batch with positive samples only during training')
parser.add_argument('--norm-scheme', default=None, help='specify how to standarize/normalize target; \'rg\':rescale grid; None: standarize')
parser.add_argument('--res', default=None, help='use standard res block structure or pixor one')
parser.add_argument('--ignorebp', action='store_true', help='flag to ignore boundary pixels')
args = parser.parse_args()

torch.manual_seed(0)

if args.model_file:
	cnf.model_file = args.model_file
	cnf.trainlog = cnf.trainlog[:-9] + args.model_file.split('/')[-1][:-11] + 'train.txt'
	cnf.trainlog2 = cnf.trainlog2[:-9] + args.model_file.split('/')[-1][:-11] + 'etime.txt'
	cnf.vallog = cnf.vallog[:-8] + args.model_file.split('/')[-1][:-11] + 'val.txt'
	cnf.gradNormlog = cnf.gradNormlog[:-9] + args.model_file.split('/')[-1][:-11] + 'gnorm.txt'
if args.root_dir:
	cnf.rootDir = args.root_dir
if args.pixor:
	args.aug_scheme = 'pixor'
elif args.voxelnet:
	args.aug_scheme = 'voxelnet'
else:
	args.aug_scheme = None
if args.epochs:
	cnf.epochs = args.epochs

# data loaders
train_loader = DataLoader(
	LidarLoader_2(cnf.rootDir+'/train', cnf.objtype, args=args, train=True),
	batch_size = cnf.batchSize, shuffle=True, num_workers=0,
	collate_fn=collate_fn_3, pin_memory=True
)
if args.val:
	val_loader = DataLoader(
		LidarLoader_2(cnf.rootDir+'/val', cnf.objtype, args=args, train=True, augData=False),
		batch_size = cnf.batchSize, shuffle=True, num_workers=0,
		collate_fn=collate_fn_3, pin_memory=True
	)

if args.standarize:
	carMean, carSTD = None, None
else:
	carMean = torch.from_numpy(cnf.carMean)
	carSTD = torch.from_numpy(cnf.carSTD)

# create detector object and intialize weights
if args.resnet18:
	hawkEye = ResNet18(mean=carMean, std=carSTD).to(cnf.device)
elif args.res == 'standard':
	hawkEye = PointCloudDetector2(cnf.res_block_layers, cnf.up_sample_layers, cnf.deconv, carMean, carSTD).to(cnf.device)
else:
	hawkEye = PointCloudDetector(cnf.res_block_layers, cnf.up_sample_layers, cnf.deconv, carMean, carSTD).to(cnf.device)
# hawkEye.apply(misc.weights_init)

if args.multi_gpu:
	hawkEye = nn.DataParallel(hawkEye)

# network optimization method
if args.step_lr:
	# optimizer = Adam(hawkEye.parameters(), lr=cnf.slr)
	optimizer = SGD(hawkEye.parameters(), lr=cnf.slr, momentum=0.9, dampening=0, weight_decay=cnf.decay, nesterov=False)
	scheduler = MultiStepLR(optimizer, milestones=cnf.milestones, gamma=0.1)
else:	
	# optimizer = Adam(hawkEye.parameters(), lr=cnf.lr)
	optimizer = SGD(hawkEye.parameters(), lr=cnf.lr, momentum=0.9, dampening=0, weight_decay=cnf.decay, nesterov=False)

# status string writer thread and queue
queue = Queue()
worker = misc.FileWriterThread(queue, cnf.trainlog)
worker.daemon = True
worker.start()

# status string writer thread and queue
queue = Queue()
worker = misc.FileWriterThread(queue, cnf.trainlog)
worker.daemon = True
worker.start()

if args.val:
	valqueue = Queue()
	valworker = misc.FileWriterThread(valqueue, cnf.vallog)
	valworker.daemon = True
	valworker.start()

def train(epoch):
	hawkEye.train()
	# empty the gradient buffer
	hawkEye.zero_grad()
	gradPresent = False

	for batchId, batch_data in enumerate(train_loader):
		st1 = time.time()
		
		data, target, filenames, zoom0_3, zoom1_2 = batch_data
		
		if data.size(0) < cnf.batchSize:
			del data
			del target
			del filenames
			del zoom0_3
			del zoom1_2
			continue

		data = data.cuda(non_blocking=True)

		# pass data through network and predict
		cla, loc = hawkEye(data)
		
		targets = []
		zoom0_3s = []
		zoom1_2s = []

		for i in range(cnf.batchSize):
			targets.append(target[i].cuda(non_blocking=True))
			zoom0_3s.append(zoom0_3[i].cuda(non_blocking=True))
			zoom1_2s.append(zoom1_2[i].cuda(non_blocking=True))

		# compute loss, gradient, and optimize
		st = time.time()
		claLoss, locLoss, posClaLoss, negClaLoss, md, meanConfidence, overallMeanConfidence, ps, ns, zr = \
			computeLoss(cla, loc, targets, zoom0_3s, zoom1_2s, args)
		ed = time.time()
		if claLoss is None:
			trainLoss = None
			tl = None
			cl = None
			ll = None
		elif locLoss is not None:
			trainLoss = claLoss + locLoss
			tl = trainLoss.item()
			cl = claLoss.item()
			ll = locLoss.item()
			posClaLoss = posClaLoss.item()
			negClaLoss = negClaLoss.item()
			gradPresent = gradPresent | True
		else:
			trainLoss = claLoss
			tl = trainLoss.item()
			cl = claLoss.item()
			ll = None
			negClaLoss = negClaLoss.item()
			gradPresent = gradPresent | True

		# trainLoss = claLoss+locLoss
		if trainLoss is not None:
			trainLoss.backward()

		# gradients are accumulated over cnf.accumulationSteps
		if (batchId+1)%cnf.accumulationSteps == 0 and gradPresent:
			gradNorm = misc.parameterNorm(hawkEye.parameters(), 'grad')
			weightNorm = misc.parameterNorm(hawkEye.parameters(), 'weight')
			misc.writeToFile(cnf.gradNormlog, cnf.normLogString.format(batchId+1, epoch+1, gradNorm, weightNorm))

			if args.clip and gradNorm > args.clipvalue:
				torch.nn.utils.clip_grad_norm_(hawkEye.parameters(), args.clipvalue)
			optimizer.step()
			hawkEye.zero_grad()

		if (batchId+1)%cnf.accumulationSteps == 0:
			gradPresent = False

		ed1 = time.time()
		queue.put((epoch+1, batchId+1, cl, negClaLoss, posClaLoss, ll, tl, int(ps), int(ns), md, meanConfidence, overallMeanConfidence, ed-st, ed1-st1, zr))

		del data
		del target
		del filenames
		del zoom0_3
		del zoom1_2

def validation(epoch):
	hawkEye.eval()

	for batchId, batch_data in enumerate(val_loader):
		st1 = time.time()
		
		data, target, filenames, zoom0_3, zoom1_2 = batch_data
		
		if data.size(0) < cnf.batchSize:
			del data
			del target
			del filenames
			del zoom0_3
			del zoom1_2
			continue

		data = data.cuda(non_blocking=True)

		# pass data through network and predict
		cla, loc = hawkEye(data)
		
		targets = []
		zoom0_3s = []
		zoom1_2s = []

		for i in range(cnf.batchSize):
			targets.append(target[i].cuda(non_blocking=True))
			zoom0_3s.append(zoom0_3[i].cuda(non_blocking=True))
			zoom1_2s.append(zoom1_2[i].cuda(non_blocking=True))

		# compute loss, gradient, and optimize
		st = time.time()
		claLoss, locLoss, posClaLoss, negClaLoss, md, meanConfidence, overallMeanConfidence, ps, ns, zr = \
			computeLoss(cla, loc, targets, zoom0_3s, zoom1_2s, args)
		ed = time.time()
		if claLoss is None:
			trainLoss = None
			tl = None
			cl = None
			ll = None
		elif locLoss is not None:
			trainLoss = claLoss + locLoss
			tl = trainLoss.item()
			cl = claLoss.item()
			ll = locLoss.item()
			posClaLoss = posClaLoss.item()
			negClaLoss = negClaLoss.item()
		else:
			trainLoss = claLoss
			tl = trainLoss.item()
			cl = claLoss.item()
			ll = None
			negClaLoss = negClaLoss.item()

		ed1 = time.time()
		valqueue.put((epoch+1, batchId+1, cl, negClaLoss, posClaLoss, ll, tl, int(ps), int(ns), md, meanConfidence, overallMeanConfidence, ed-st, ed1-st1, zr))

		del data
		del target
		del filenames
		del zoom0_3
		del zoom1_2

if __name__ == '__main__':
	# load model file if present
	if os.path.isfile(cnf.model_file):
		hawkEye.load_state_dict(torch.load(cnf.model_file,
			map_location=lambda storage, loc: storage))

	try:
		for epoch in range(cnf.epochs):
			# learning rate decay scheduler
			if args.step_lr:
				scheduler.step()

			st = time.time()
			train(epoch)
			ed = time.time()
			misc.writeToFile(cnf.trainlog2, '~~~~~epoch ' + str(epoch) + ' end time taken: '+str(ed-st)+' secs~~~~\n')

			# run validation every 10 epochs
			if args.val and (epoch+1)%5==0:
				validation(epoch)

			if (epoch+1)%3 == 0:
				torch.save(hawkEye.state_dict(), cnf.model_file)

	except BaseException as e:
		trace = traceback.format_exc()
		misc.writeToFile(cnf.errorlog, trace+'\n\n\n')
	finally:
		torch.save(hawkEye.state_dict(), cnf.model_file)
		del hawkEye

	# finish all tasks
	queue.join()

	if args.val:
		valqueue.join()