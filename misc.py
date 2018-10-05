import torch
import os

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
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)
		# concatenate the tensors [cla, loc]
		frameClaLoc = torch.cat((frameCla, frameLoc), 1)

		# save
		filename = filename.split('/')[-1][:-4]

		# make directory if it doesnt exists
		if not os.path.exists(outputDir+'/'+str(epoch)):
			os.makedirs(outputDir+'/'+str(epoch))

		torch.save(frameClaLoc,
			outputDir+'/'+str(epoch)+'/'+filename+'.pt')

def writeToFile(filename, line):
	with open(filename, 'a') as file:
		file.write(line)