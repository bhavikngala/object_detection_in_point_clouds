import os
import shutil

def splitTrainValiAccorToImagesets():
	lidarRootDir = './../training/velodyne'
	labelRootDir = './../training (2)/label_2'

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


def trainVali9010Split():
	trainFiles = './../data/KITTI_BEV/train'
	valFiles = './../data/KITTI_BEV/val'
	eightySplit = './../data/KITTI_BEV/9010'

	tf = [f for f in os.listdir(trainFiles) if os.path.isfile(os.path.join(trainFiles, f))]
	vf = [f for f in os.listdir(valFiles) if os.path.isfile(os.path.join(valFiles, f))]

	t = len(tf) + len(vf)
	v = int(0.1 * t)

	for i in range(v):
		f = vf.pop(0)
		shutil.copy(os.path.join(valFiles, f), os.path.join(eightySplit, 'val', f))
		shutil.copy(os.path.join(valFiles, 'labels', f[:-4]+'.txt'), os.path.join(eightySplit, 'val', 'labels', f[:-4]+'.txt'))
		print('val', f)

	while vf:
		f = vf.pop(0)
		shutil.copy(os.path.join(valFiles, f), os.path.join(eightySplit, 'train', f))
		shutil.copy(os.path.join(valFiles, 'labels', f[:-4]+'.txt'), os.path.join(eightySplit, 'train', 'labels', f[:-4]+'.txt'))
		print('train', f)

	while tf:
		f = tf.pop(0)
		shutil.copy(os.path.join(trainFiles, f), os.path.join(eightySplit, 'train', f))
		shutil.copy(os.path.join(trainFiles, 'labels', f[:-4]+'.txt'), os.path.join(eightySplit, 'train', 'labels', f[:-4]+'.txt'))
		print('train', f)

	print(len(vf), len(tf))
	print(v, t-v)

if __name__=='__main__':
	trainVali9010Split()