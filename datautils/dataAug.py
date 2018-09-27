import numpy as np
from numpy import random

from os import listdir
from os.path import isfile, join

trainDir = './../../data_object_velodyne/training/velodyne'
labelDir = './../../data/KITTI_BEV/training/label_2'

filenames = [join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))]
labels = [join(labelDir, f) for f in listdir(labelDir) if isfile(join(labelDir, f))]

# randomly select a tranformation for each file
# 0: rotation
# 1: global scaling
# 2: reflect around y axis
transformation = random.randint(low=0, high=3, size=len(filenames))

for (lidarFile, labelFile, t) in zip(filenames, labels, transformation):
	theta = None
	tmat = None

	print(labelFile)
	# read lidar file
	lidar = np.fromfile(lidarFile,
		dtype=np.float32).reshape(-1, 4)

	labels = []
	# read labels
	with open(labelFile) as f:
		line = f.readline()
		while(line):
			labels.append(line)
			line = f.readline()

	if t == 0:
		# rotation
		theta = random.uniform(low=-np.pi/4, high=(np.pi/4+np.pi/180))
		tmat = np.array([[np.cos(theta), -np.sin(theta), 0],
						 [np.sin(theta),  np.cos(theta), 0],
						 [            0,              0, 1]])
	elif t == 1:
		# scaling
		scale = random.uniform(low=0.95, high=1.06)
		tmat = np.array([[scale,     0,     0],
						 [    0, scale,     0],
						 [    0,     0, scale]])
	else:
		# reflect y axis
		tmat = np.array([[1,  0, 0],
						 [0, -1, 0],
						 [0,  0, 1]])

	# transform lidar data
	lidar[:,:3] = np.matmul(tmat, lidar[:,:3].T).T

	tlabels = []
	for line in labels:
		data = line.split()
		xyz = np.array([[float(data[11]), float(data[12]), float(data[13])]])
		xyz = np.matmul(tmat, xyz.T)
		data[11] = str(xyz[0][0])
		data[12] = str(xyz[1][0])
		data[13] = str(xyz[2][0])

		if theta:
			data[3] = str(float(data[3]) - theta)

		tlabels.append(' '.join(data))

	augFileName = lidarFile.split('/')[-1][:-4]
	np.save(trainDir+'/'+augFileName+'_'+str(t)+'.npy', lidar)

	augLabelFilename = labelFile.split('/')[-1][:-4]
	with open(labelDir+'/'+augLabelFilename+'_'+str(t)+'.txt', 'w') as f:
		for line in tlabels:
			f.write(line+'\n')