import os
import shutil

lidarRootDir = './../training/velodyne'
labelRootDir = './../training/label_2'

destDir = './../data/KITTI_BEV'

lidarFilenames = [f for f in os.listdir(lidarRootDir) if os.path.isfile(os.path.join(lidarRootDir, f))]

trainFiles = []
valiFiles = []

with open('./../ImageSets/train.txt', 'r') as f:
	line = f.readline().strip()
	while line:
		trainFiles.append(line)
		line = f.readline().strip()

with open('./../ImageSets/val.txt', 'r') as f:
	line = f.readline().strip()
	while line:
		valiFiles.append(line)
		line = f.readline().strip()

for filename in lidarFilenames:
	filename1 = filename[:-4].split('_')[0]

	if filename1 in trainFiles:
		# print(destDir+'/train/'+filename)
		shutil.move(lidarRootDir+'/'+filename, destDir+'/train/'+filename)
		shutil.move(labelRootDir+'/'+filename1+'.txt', destDir+'/train/labels/'+filename1+'.txt')

	elif filename1 in valiFiles:
		# print(destDir+'/val/'+filename)
		shutil.move(lidarRootDir+'/'+filename, destDir+'/val/'+filename)
		shutil.move(labelRootDir+'/'+filename1+'.txt', destDir+'/val/labels/'+filename1+'.txt')