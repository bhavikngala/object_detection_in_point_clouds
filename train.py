import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import os
import time
from queue import Queue

from networks.networks import PointCloudDetector as HawkEye
from datautils.dataloader import *
from datautils.DataLoaderThread import DataLoaderThread
import config as cnf
from lossUtils import computeLoss
import misc

torch.manual_seed(0)

# data loaders
train_loader = DataLoader(
	LidarLoader_2(cnf.rootDir),
	batch_size = cnf.batchSize, shuffle=True, num_workers=3,
	collate_fn=collate_fn_2, pin_memory=True
)

# create detector object and intialize weights
hawkEye = HawkEye(cnf.res_block_layers, cnf.up_sample_layers).to(cnf.device)
hawkEye.apply(misc.weights_init)

# network optimization method
optimizer = Adam(hawkEye.parameters(), lr=cnf.learningRate)
scheduler = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

# status string writier thread and queue
queue = Queue()
worker = misc.FileWriterThread(queue, cnf.trainlog)
worker.daemon = True
worker.start()


def train(epoch):
	hawkEye.train()

	for batchId, batch_data in enumerate(train_loader):
		st1 = time.time()
		# empty the gradient buffer
		hawkEye.zero_grad()
		
		data, target, filenames = batch_data

		data = data.cuda(device=None, non_blocking=True)
		target = target.cuda(device=None, non_blocking=True)
		m, tr, tc = target.size()
		# create zoom boxes
		zoom0_3 = target.new_full([m, tr, 4], fill_value=0)
		zoom1_2 = target.new_full([m, tr, 4], fill_value=0)

		# left: y + w/2, right: y - w/2, forward: x + l/2, backward: x - l/2
		zoom1_2[:, 0] = labels[:, 4] + labels[:, 6]*0.6
		zoom1_2[:, 1] = labels[:, 4] - labels[:, 6]*0.6
		zoom1_2[:, 2] = labels[:, 3] + labels[:, 5]*0.6
		zoom1_2[:, 3] = labels[:, 3] - labels[:, 5]*0.6

		zoom0_3[:, 0] = labels[:, 4] + labels[:, 6]*0.15
		zoom0_3[:, 1] = labels[:, 4] - labels[:, 6]*0.15
		zoom0_3[:, 2] = labels[:, 3] + labels[:, 5]*0.15
		zoom0_3[:, 3] = labels[:, 3] - labels[:, 5]*0.15

		# pass data through network and predict
		cla, loc = hawkEye(data)

		# compute loss, gradient, and optimize
		st = time.time()
		claLoss, locLoss = computeLoss(cla, loc, target, zoom0_3, zoom1_2)
		ed = time.time()
		if claLoss is None:
			trainLoss = None
			# ls = cnf.logString3.format(epoch, batchId)
		elif locLoss is not None:
			trainLoss = claLoss + locLoss
			# ls = cnf.logString1.format(epoch, batchId, claLoss.item(), locLoss.item(), trainLoss.item())
		else:
			trainLoss = claLoss
			# ls = cnf.logString2.format(epoch, batchId, claLoss.item(), trainLoss.item())

		# trainLoss = claLoss+locLoss
		if trainLoss is not None:
			trainLoss.backward()
			optimizer.step()

		# TODO: mAP

		# save the results, loss in a file
		if (epoch+1)==cnf.epochs:
			misc.savebatchOutput(cla, loc, filenames, cnf.trainOutputDir, epoch)
			misc.savebatchTarget(target, filenames, cnf.trainOutputDir, epoch)
		
		ed1 = time.time()
		# ls = ls + 'elapsed time: '+str(ed-st)+' secs ' + 'batch elapsed time: '+str(ed1-st1)+' secs\n\n'
		queue.put((epoch, i, claLoss, locLoss, trainLoss, (ed-st), (ed1-st1)))

def validation(epoch):
	hawkEye.eval()

	for batchId, batch_data in enumerate(vali_loader):
		data, target, filenames, zoom0_3, zoom1_2 = batch_data

		# move data to GPU
		data = data.to(cnf.device)
		target = target.to(cnf.device)
		zoom1_2 = zoom1_2.to(cnf.device)
		zoom0_3 = zoom0_3.to(cnf.device)

		# pass data through network and predict
		cla, loc = hawkEye(data)

		claLoss, locLoss = computeLoss(cla, loc, target, zoom0_3, zoom1_2)
		if claLoss is None:
			valLoss = None
			ls = cnf.logString3.format(epoch, batchId)
		elif locLoss is not None:
			valLoss = claLoss + locLoss
			ls = cnf.logString1.format(epoch, batchId, claLoss.item(), locLoss.item(), valLoss.item())
		else:
			valLoss = claLoss
			ls = cnf.logString2.format(epoch, batchId, claLoss.item(), valLoss.item())


		# TODO mAP

		# save the results, loss in a file
		if (epoch+1)==cnf.epochs:
			misc.savebatchOutput(cla, loc, filenames, cnf.valiOutputDir, epoch)
			misc.savebatchTarget(target, filenames, cnf.valiOutputDir, epoch)
		
		queue.put((cnf.vallog, ls))
		# misc.writeToFile(cnf.vallog, ls)

if __name__ == '__main__':
	# load model file if present
	if os.path.isfile(cnf.model_file):
		hawkEye.load_state_dict(torch.load(cnf.model_file,
			map_location=lambda storage, loc: storage))

	try:
		for epoch in range(cnf.epochs):
			# learning rate decay scheduler
			scheduler.step()

			st = time.time()
			train(epoch)
			ed = time.time()
			misc.writeToFile(cnf.trainlog, '\n\n\n~~~~~epoch end time taken: '+str(st-ed)+' secs~~~~\n\n\n')

			# run validation every 10 epochs
			# if (epoch+1)%10 == 0:
			# 	validation(epoch)

			if (epoch+1)%10 == 0:
				torch.save(hawkEye.state_dict(), cnf.model_file)
		finally:
			torch.save(hawkEye.state_dict(), cnf.model_file)

	# finish all tasks
	queue.join()