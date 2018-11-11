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
				epoch, batchId, cl, ll, tl, objs, ps, ns, iou, mc, lt, bt = self.queue.get()
				if cl is None:
					ls = cnf.logString3.format(epoch, batchId, objs, ps, ns, lt, bt)
				elif ll is not None:
					ls = cnf.logString1.format(epoch, batchId, cl, ll, tl, objs, ps, ns, iou, mc, lt, bt)
				else:
					ls = cnf.logString2.format(epoch, batchId, cl, tl, objs, ps, ns, lt, bt)
				writeToFile(self.filename, ls)
			finally:
				self.queue.task_done()