import torch
import numpy as np
import config as cnf

# source: https://github.com/jeasinema/VoxelNet-tensorflow/blob/master/utils/utils.py

def angle_in_limit(angle):
	# To limit the angle in -pi/2 - pi/2
	limit_degree = 5
	while angle >= np.pi / 2:
		angle -= np.pi
	while angle < -np.pi / 2:
		angle += np.pi
	if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
		angle = np.pi / 2
	return angle


def camera_to_lidar(x, y, z):
	p = np.array([x, y, z, 1])
	p = np.matmul(np.linalg.inv(np.array(cnf.MATRIX_R_RECT_0)), p)
	p = np.matmul(np.linalg.inv(np.array(cnf.MATRIX_T_VELO_2_CAM)), p)
	p = p[0:3]
	return tuple(p)


def lidar_to_camera(x, y, z):
	p = np.array([x, y, z, 1])
	p = np.matmul(np.array(cnf.MATRIX_T_VELO_2_CAM), p)
	p = np.matmul(np.array(cnf.MATRIX_R_RECT_0), p)
	p = p[0:3]
	return tuple(p)


def camera_to_lidar_point(points):
	# (N, 3) -> (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

	points = np.matmul(np.linalg.inv(np.array(cnf.MATRIX_R_RECT_0)), points)
	points = np.matmul(np.linalg.inv(
		np.array(cnf.MATRIX_T_VELO_2_CAM)), points).T  # (4, N) -> (N, 4)
	points = points[:, 0:3]
	return points.reshape(-1, 3)


def lidar_to_camera_point(points):
	# (N, 3) -> (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))]).T

	points = np.matmul(np.array(cnf.MATRIX_T_VELO_2_CAM), points)
	points = np.matmul(np.array(cnf.MATRIX_R_RECT_0), points).T
	points = points[:, 0:3]
	return points.reshape(-1, 3)


def camera_to_lidar_box(boxes):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, ry = box
		(x, y, z), h, w, l, rz = camera_to_lidar(x, y, z), \
			h, w, l, -ry - np.pi / 2
		rz = angle_in_limit(rz)
		ret.append([x, y, z, h, w, l, rz])
	return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, rz = box
		(x, y, z), h, w, l, ry = lidar_to_camera(x, y, z), \
			h, w, l, -rz - np.pi / 2
		ry = angle_in_limit(ry)
		ret.append([x, y, z, h, w, l, ry])
	return np.array(ret).reshape(-1, 7)