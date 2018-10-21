import torch
import os
from threading import Thread
import config as cnf

# custom weights initialization called on network
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
	# m.bias.data.fill_(0)

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

class FileWriterThread(Thread):

	def __init__(self, queue, filename):
		Thread.__init__(self)
		self.queue = queue
		self.filename = filename

	def run(self):
		while True:
			try:
				epoch, batchId, claLoss, locLoss, trainLoss, lt, bt, ps, ns = self.queue.get()
				if claLoss is None:
					trainLoss = None
					ls = cnf.logString3.format(epoch, batchId, lt, bt, ps, ns)
				elif locLoss is not None:
					trainLoss = claLoss + locLoss
					ls = cnf.logString1.format(epoch, batchId, claLoss.item(), locLoss.item(), trainLoss.item(), lt, bt, ps, ns)
				else:
					trainLoss = claLoss
					ls = cnf.logString2.format(epoch, batchId, claLoss.item(), trainLoss.item(), lt, bt, ps, ns)
				writeToFile(self.filename, ls)
			finally:
				self.queue.task_done()