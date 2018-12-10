import numpy as np
import os


class KittiReader():

	def __init__(self, dirList):
		self.lidarDir = dirList[0]
		self.labelsDir = dirList[1]
		self.calibDir = dirList[2]
		self.leftColorImageDir = dirList[3]

	def readLidarFile(self, filename):
		if self.lidarDir is None or filename is None:
			return None
		file = os.path.join(self.lidarDir, filename+'.bin')
		lidar = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
		return lidar

	def readLabels(self, filename):
		if self.labelsDir is None or filename is None:
			return None
		file = os.path.join(self.labelsDir, filename+'.txt')
		lines = [line.rstrip().lower().split() for line in open(file)]
		lines = np.array(lines)
		return lines

	def readCalibrationDict(self, filename):
		# Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
		if self.calibDir is None or filename is None:
			return None
		file = os.path.join(self.calibDir, filename+'.txt')
		data = {}
		with open(file, 'r') as f:
			for line in f.readlines():
				line = line.rstrip()
				if len(line)==0: continue
				key, value = line.split(':', 1)
				# The only non-float values in these files are dates, which
				# we don't care about anyway
				try:
					data[key] = np.array([float(x) for x in value.split()])
				except ValueError:
					pass

		return data

	def readLeftColorImage(self, filename):
		if self.leftColorImageDir is None or filename is None:
			return None
		file = os.path.join(self.leftColorImageDir, filename+'.png')
		img = cv2.imread(file)
		return img


class ProjectKittiToDifferentCoordinateSystems():
	# reference: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py

	def __init__(self):
		self.P2 = None
		self.R0 = None
		self.V2C = None
		self.C2V = None

	def setCalibrationMatrices(self, calibDict):
		self.P2 = calibDict['P2'].np.reshape(self.P, [3,4])
		self.R0 = calibDict['R0_rect'].np.reshape(self.R0,[3,3])
		self.V2C = calibDict['Tr_velo_to_cam'].np.reshape(self.V2C, [3,4])
		self.C2V = inverse_rigid_trans(V2C)

	def clearCalibrationMatrices(self):
		self.P2 = None
		self.R0 = None
		self.V2C = None
		self.C2V = None

	def cart2hom(self, pts_3d):
		''' Input: nx3 points in Cartesian
			Oupput: nx4 points in Homogeneous by pending 1
		'''
		n = pts_3d.shape[0]
		pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
		return pts_3d_hom

	# =========================== 
	# ------- 3d to 3d ---------- 
	# =========================== 
	def project_velo_to_ref(self, pts_3d_velo):
		pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
		return np.dot(pts_3d_velo, np.transpose(self.V2C))

	def project_ref_to_velo(self, pts_3d_ref):
		pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
		return np.dot(pts_3d_ref, np.transpose(self.C2V))

	def project_rect_to_ref(self, pts_3d_rect):
		''' Input and Output are nx3 points '''
		return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
	
	def project_ref_to_rect(self, pts_3d_ref):
		''' Input and Output are nx3 points '''
		return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
	def project_rect_to_velo(self, pts_3d_rect):
		''' Input: nx3 points in rect camera coord.
			Output: nx3 points in velodyne coord.
		''' 
		pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
		return self.project_ref_to_velo(pts_3d_ref)

	def project_velo_to_rect(self, pts_3d_velo):
		pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
		return self.project_ref_to_rect(pts_3d_ref)

	# =========================== 
	# ------- 3d to 2d ---------- 
	# =========================== 
	def project_rect_to_image(self, pts_3d_rect):
		''' Input: nx3 points in rect camera coord.
			Output: nx2 points in image2 coord.
		'''
		pts_3d_rect = self.cart2hom(pts_3d_rect)
		pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
		pts_2d[:,0] /= pts_2d[:,2]
		pts_2d[:,1] /= pts_2d[:,2]
		return pts_2d[:,0:2]
	
	def project_velo_to_image(self, pts_3d_velo):
		''' Input: nx3 points in velodyne coord.
			Output: nx2 points in image2 coord.
		'''
		pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
		return self.project_rect_to_image(pts_3d_rect)

	# =========================== 
	# ------- 2d to 3d ---------- 
	# =========================== 
	def project_image_to_rect(self, uv_depth):
		''' Input: nx3 first two channels are uv, 3rd channel
				   is depth in rect camera coord.
			Output: nx3 points in rect camera coord.
		'''
		n = uv_depth.shape[0]
		x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
		y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
		pts_3d_rect = np.zeros((n,3))
		pts_3d_rect[:,0] = x
		pts_3d_rect[:,1] = y
		pts_3d_rect[:,2] = uv_depth[:,2]
		return pts_3d_rect

	def project_image_to_velo(self, uv_depth):
		pts_3d_rect = self.project_image_to_rect(uv_depth)
		return self.project_rect_to_velo(pts_3d_rect)

def rotx(t):
	''' 3D Rotation about the x-axis. '''
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[1,  0,  0],
					 [0,  c, -s],
					 [0,  s,  c]])


def roty(t):
	''' Rotation about the y-axis. '''
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c,  0,  s],
					 [0,  1,  0],
					 [-s, 0,  c]])


def rotz(t):
	''' Rotation about the z-axis. '''
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c, -s,  0],
					 [s,  c,  0],
					 [0,  0,  1]])


def transform_from_rot_trans(R, t):
	''' Transforation matrix from rotation matrix and translation vector. '''
	R = R.reshape(3, 3)
	t = t.reshape(3, 1)
	return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
	''' Inverse a rigid body transform matrix (3x4 as [R|t])
		[R'|-R't; 0|1]
	'''
	inv_Tr = np.zeros_like(Tr) # 3x4
	inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
	inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
	return inv_Tr