import torch
import torch.nn as nn
import os
from threading import Thread
from queue import Queue
import config as cnf


def savebatchOutput(cla, loc, filenames, outputDir, epoch):
	for i in range(len(filenames)):
		filename = filenames[i]

		# make directory if it doesnt exists
		if not os.path.exists(outputDir+'/'+str(epoch)+'/output'):
			os.makedirs(outputDir+'/'+str(epoch)+'/output')

		torch.save(cla[i],
			outputDir+'/'+str(epoch)+'/output/'+filename+'_cla.pt')
		torch.save(loc[i],
			outputDir+'/'+str(epoch)+'/output/'+filename+'_loc.pt')


def savebatchTarget(target, filenames, outputDir, epoch):
	for i in range(len(filenames)):
		filename = filenames[i]

		# make directory if it doesnt exists
		if not os.path.exists(outputDir+'/'+str(epoch)+'/target'):
			os.makedirs(outputDir+'/'+str(epoch)+'/target')

		torch.save(target[i],
			outputDir+'/'+str(epoch)+'/target/'+filename+'.pt')


def writeToFile(filename, line):
	with open(filename, 'a') as file:
		file.write(line)


class Logger():

	def __init__(self, filename):
		queue = Queue()
		worker = FileWriterThread(queue, cnf.trainlog)
		worker.daemon = True
		worker.start()

	def join(self):
		self.queue.join()

	def put(self, s):
		self.queue.put(s)


class FileWriterThread(Thread):

	def __init__(self, queue, filename):
		Thread.__init__(self)
		self.queue = queue
		self.filename = filename

	def run(self):
		while True:
			try:
				epoch, batchId, cl, nsl, psl, ll, tl, ps, ns, md, mc, oamc, lt, bt = self.queue.get()
				if cl is None:
					ls = cnf.logString3.format(batchId, epoch, ps, ns, lt, bt)
				elif ll is not None:
					ls = cnf.logString1.format(batchId, epoch, cl, nsl, psl, ll, tl, ps, ns, md, mc, oamc, lt, bt)
				else:
					ls = cnf.logString2.format(batchId, epoch, cl, nsl, tl, ps, ns, oamc, lt, bt)
				writeToFile(self.filename, ls)
			finally:
				self.queue.task_done()


def parameterNorm(parameters, p):
	vec = []
	for param in parameters:
		if p == 'grad':
			vec.append(param.grad.view(-1))
		elif p == 'weight':
			vec.append(param.view(-1))
	param_norm = torch.cat(vec).norm(2)
	return param_norm