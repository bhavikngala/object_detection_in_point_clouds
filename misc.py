import torch
import os
from threading import Thread
from queue import Queue
import argparse
import config as cnf


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


def getArgumentParser():
	parser = argparse.ArgumentParser(description='Train network')
	args = parser.parse_args()

	return args