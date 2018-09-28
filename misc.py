import torch

def savebatchOutput(cla, loc, filenames, outputDir, epoch):
	for i in range(filenames.shape[0]):
		filename = filenames[i]
		frameCla = cla[i].view(-1, 1)
		frameLoc = loc[i].view(-1, 6)
		# concatenate the tensors [cla, loc]
		frameClaLoc = torch.cat((frameCla, frameLoc), 1)

		# save
		filename = filename.split('/')[-1][:-4]

		torch.save(frameClaLoc,
			outputDir+'/'+str(epoch)+'/'+filename+'.pt')

def writeToFile(filename, line):
	with open(filename, 'a') as file:
		file.write(line)