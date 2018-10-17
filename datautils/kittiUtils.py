import numpy as np
import config as cnf

'''
References:
1. http://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
2. https://github.com/Mi-lo/pykitti
'''

def readR0AndTRVeloToCamMatrices(filename):
	'''
	Reads R0 and TR_velo_to_cam matrices from calibration file
	params : Filename where the calibration data is store
	returns: numpy arrays of R0 and Tr_velo_to_cam 
	'''
	d = {}
	with open(filename) as f:
		for line in f.readlines():
			key, value = line.split(':', 1)
			d[key] = np.array([float(v) for v in value])

	R0, Tr_velo_to_cam = np.zeros((4,4)), np.zeros((4,4))
	R0[3,3], Tr_velo_to_cam[3,3] = 1
	
	R0[:3, :3] = d['R0'].reshape(3,3)
	Tr_velo_to_cam[:3, :] = d['Tr_velo_to_cam'].reshape(3, 4)

	return R0, Tr_velo_to_cam


def convertCamera0ToLidar(camera0Points, calibFilename):
	'''
	Converts 3D points in lidar cordinate system to 3D points in 
	Camera 0 cordinate system
	params : np.array, points in lidar system, (N, 7), (x, y, z, h, w, l, ry)
	params : string, filename of the calibration data
	returns: np.array, points in camera0 system, (N, 7), (x', y', z', h, w, l, ry')
	'''
	R0, Tr_velo_to_cam = readR0AndTRVeloToCamMatrices(calibFilename)
	# take inverse of the matrices
	R0_inv = np.linalg.inv(R0)
	Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)

	camPoints = np.ones((camera0Points.shape[0], 4))
	camPoints[:, :3] = camera0Points[:,:3]

	lidarPoints = np.matmul(R0_inv, np.matmul(Tr_velo_to_cam_inv, camPoints.T))

	# convert from homogenous points to normal points
	lidarPoints[:3,:] = lidarPoints[:3,:]/lidarPoints[3,:]
	
	camera0Points[:,:3] = lidarPoints.T[:,:3]
	return camera0Points