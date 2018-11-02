import numpy as np
import math
import cv2
import config as cnf


# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : Thu 08 Mar 2018 02:30:56 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]
# Source : https://github.com/jeasinema/VoxelNet-tensorflow/blob/master/utils/utils.py


def lidar_to_bird_view(x, y, factor=1):
	x_r, y_r, z_r = cnf.gridConfig['x'], cnf.gridConfig['y'], cnf.gridConfig['z']
	res = cnf.gridConfig['res']
	a = -(y + int(y_r[0])/res)
	b = x/res
	a = np.clip(a, a_max=(x_r[1] - x_r[0]) / res, a_min=0)
	b = np.clip(b, a_max=(y_r[1] - y_r[0]) / res, a_min=0)
	return a, b


def batch_lidar_to_bird_view(points, factor=1):
	# Input:
	#   points (N, 2)
	# Outputs:
	#   points (N, 2)
	x_r, y_r, z_r = cnf.gridConfig['x'], cnf.gridConfig['y'], cnf.gridConfig['z']
	res = cnf.gridConfig['res']
	a = -(points[:, 1] + y_r[0]) / res
	b = points[:, 0]/res
	a = np.clip(a, a_max=(x_r[1] - x_r[0]) / res, a_min=0)
	b = np.clip(b, a_max=(y_r[1] - y_r[0]) / res, a_min=0)
	return np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis=-1)


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
	p = np.matmul(cnf.R0_inv, p)
	p = np.matmul(cnf.Tr_velo_to_cam_inv, p)
	p = p[0:3]
	return tuple(p)


def lidar_to_camera(x, y, z):
	p = np.array([x, y, z, 1])
	p = np.matmul(cnf.Tr_velo_to_cam, p)
	p = np.matmul(cnf.R0, p)
	p = p[0:3]
	return tuple(p)


def camera_to_lidar_point(points):
	# (N, 3) -> (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

	points = np.matmul(cnf.R0_inv, points)
	points = np.matmul(cnf.Tr_velo_to_cam_inv, points).T  # (4, N) -> (N, 4)
	points = points[:, 0:3]
	return points.reshape(-1, 3)


def lidar_to_camera_point(points):
	# (N, 3) -> (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))]).T

	points = np.matmul(cnf.Tr_velo_to_cam, points)
	points = np.matmul(cnf.R0, points).T
	points = points[:, 0:3]
	return points.reshape(-1, 3)


def camera_to_lidar_box(boxes):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, ry = box
		(x, y, z), h, w, l, rz = camera_to_lidar(
			x, y, z), h, w, l, -ry - np.pi / 2
		rz = angle_in_limit(rz)
		ret.append([x, y, z, h, w, l, rz])
	return np.array(ret).reshape(-1, 7)


def camera_to_lidar_box_1(boxes):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for i in range(boxes.shape[0]):
		x, y, z, h, w, l, ry = boxes[i]
		(x, y, z), h, w, l, rz = camera_to_lidar(
			x, y, z), h, w, l, -ry - np.pi / 2
		rz = angle_in_limit(rz)
		ret.append([x, y, z, h, w, l, rz])
	return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, rz = box
		(x, y, z), h, w, l, ry = lidar_to_camera(
			x, y, z), h, w, l, -rz - np.pi / 2
		ry = angle_in_limit(ry)
		ret.append([x, y, z, h, w, l, ry])
	return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center, coordinate='lidar'):
	# (N, 5) -> (N, 4, 2)
	N = boxes_center.shape[0]
	boxes3d_center = np.zeros((N, 7))
	boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
	boxes3d_corner = center_to_corner_box3d(
		boxes3d_center, coordinate=coordinate)

	return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate='lidar'):
	# (N, 7) -> (N, 8, 3)
	N = boxes_center.shape[0]
	ret = np.zeros((N, 8, 3), dtype=np.float32)

	if coordinate == 'camera':
		boxes_center = camera_to_lidar_box(boxes_center)

	for i in range(N):
		box = boxes_center[i]
		translation = box[0:3]
		size = box[3:6]
		rotation = [0, 0, box[-1]]

		h, w, l = size[0], size[1], size[2]
		trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
			[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
			[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
			[0, 0, 0, 0, h, h, h, h]])

		# re-create 3D bounding box in velodyne coordinate system
		yaw = rotation[2]
		rotMat = np.array([
			[np.cos(yaw), -np.sin(yaw), 0.0],
			[np.sin(yaw), np.cos(yaw), 0.0],
			[0.0, 0.0, 1.0]])
		cornerPosInVelo = np.dot(rotMat, trackletBox) + \
			np.tile(translation, (8, 1)).T
		box3d = cornerPosInVelo.transpose()
		ret[i] = box3d

	if coordinate == 'camera':
		for idx in range(len(ret)):
			ret[idx] = lidar_to_camera_point(ret[idx])

	return ret


def corner_to_center_box2d(boxes_corner, coordinate='lidar'):
	# (N, 4, 2) -> (N, 5)  x,y,w,l,r
	N = boxes_corner.shape[0]
	boxes3d_corner = np.zeros((N, 8, 3))
	boxes3d_corner[:, 0:4, 0:2] = boxes_corner
	boxes3d_corner[:, 4:8, 0:2] = boxes_corner
	boxes3d_center = corner_to_center_box3d(
		boxes3d_corner, coordinate=coordinate)

	return boxes3d_center[:, [0, 1, 4, 5, 6]]


def corner_to_standup_box2d(boxes_corner):
	# (N, 4, 2) -> (N, 4) x1, y1, x2, y2
	N = boxes_corner.shape[0]
	standup_boxes2d = np.zeros((N, 4))
	standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
	standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
	standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
	standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

	return standup_boxes2d


# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
	# (N, 4) -> (N, 4) x,y,w,l -> x1,y1,x2,y2
	anchor_standup = np.zeros_like(anchors)
	# r == 0
	anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
	anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
	anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
	anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
	# r == pi/2
	anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
	anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
	anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
	anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

	return anchor_standup

CORNER2CENTER_AVG = True
def corner_to_center_box3d(boxes_corner, coordinate='camera'):
	# (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
	if coordinate == 'lidar':
		for idx in range(len(boxes_corner)):
			boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx])
	ret = []
	for roi in boxes_corner:
		if CORNER2CENTER_AVG:  # average version
			roi = np.array(roi)
			h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
			w = np.sum(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
			) / 4
			l = np.sum(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
			) / 4
			x = np.sum(roi[:, 0], axis=0) / 8
			y = np.sum(roi[0:4, 1], axis=0) / 4
			z = np.sum(roi[:, 2], axis=0) / 8
			ry = np.sum(
				math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
				math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
				math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
				math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
				math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
				math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
				math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
				math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
			) / 8
			if w > l:
				w, l = l, w
				ry = angle_in_limit(ry + np.pi / 2)
		else:  # max version
			h = max(abs(roi[:4, 1] - roi[4:, 1]))
			w = np.max(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
			)
			l = np.max(
				np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
				np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
			)
			x = np.sum(roi[:, 0], axis=0) / 8
			y = np.sum(roi[0:4, 1], axis=0) / 4
			z = np.sum(roi[:, 2], axis=0) / 8
			ry = np.sum(
				math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
				math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
				math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
				math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
				math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
				math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
				math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
				math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
			) / 8
			if w > l:
				w, l = l, w
				ry = angle_in_limit(ry + np.pi / 2)
		ret.append([x, y, z, h, w, l, ry])
	if coordinate == 'lidar':
		ret = camera_to_lidar_box(np.array(ret))

	return np.array(ret)


# this just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False):
	# (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
	num = len(boxes3d)
	boxes2d = np.zeros((num, 4), dtype=np.int32)
	projections = np.zeros((num, 8, 2), dtype=np.float32)

	lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate='lidar')

	for n in range(num):
		box3d = lidar_boxes3d_corner[n]
		box3d = lidar_to_camera_point(box3d)
		points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
		points = np.matmul(cnf.P2, points).T
		points[:, 0] /= points[:, 2]
		points[:, 1] /= points[:, 2]

		projections[n] = points[:, 0:2]
		minx = int(np.min(points[:, 0]))
		maxx = int(np.max(points[:, 0]))
		miny = int(np.min(points[:, 1]))
		maxy = int(np.max(points[:, 1]))

		boxes2d[n, :] = minx, miny, maxx, maxy

	return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, factor=1):
	# Input:
	#   lidar: (N', 4)
	# Output:
	#   birdview: (w, l, 3)
	# birdview = np.zeros(
	# 	(cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
	# for point in lidar:
	# 	x, y = point[0:2]
	# 	if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
	# 		x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
	# 				   factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
	# 		birdview[y, x] += 1
	# birdview = birdview - np.min(birdview)
	# divisor = np.max(birdview) - np.min(birdview)
	# # TODO: adjust this factor
	# birdview = np.clip((birdview / divisor * 255) *
	# 				   5 * factor, a_min=0, a_max=255)
	# birdview = np.tile(birdview, 3).astype(np.uint8)

	return None


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]),
							  color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1):
	# Input:
	#   img: (h, w, 3)
	#   boxes3d (N, 7) [x, y, z, h, w, l, r]
	#   scores
	#   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
	img = img.copy()
	projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True)
	gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True)

	# draw projections
	for qs in projections:
		for k in range(0, 4):
			i, j = k, (k + 1) % 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), color, thickness, cv2.LINE_AA)

			i, j = k + 4, (k + 1) % 4 + 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), color, thickness, cv2.LINE_AA)

			i, j = k, k + 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), color, thickness, cv2.LINE_AA)

	# draw gt projections
	for qs in gt_projections:
		for k in range(0, 4):
			i, j = k, (k + 1) % 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

			i, j = k + 4, (k + 1) % 4 + 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

			i, j = k, k + 4
			cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
												 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

	return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=np.array([]),
								 color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, factor=1):
	# Input:
	#   birdview: (h, w, 3)
	#   boxes3d (N, 7) [x, y, z, h, w, l, r]
	#   scores
	#   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
	img = birdview.copy()
	corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar')
	corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar')
	# draw gt
	for box in corner_gt_boxes3d:
		x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
		x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
		x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
		x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

		cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
				 gt_color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
				 gt_color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
				 gt_color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
				 gt_color, thickness, cv2.LINE_AA)

	# draw detections
	for box in corner_boxes3d:
		x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
		x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
		x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
		x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

		cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
				 color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
				 color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
				 color, thickness, cv2.LINE_AA)
		cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
				 color, thickness, cv2.LINE_AA)

	return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def label_to_gt_box3d(labels, cls='Car', coordinate='camera'):
	# Input:
	#   label: (N, N')
	#   cls: 'Car' or 'Pedestrain' or 'Cyclist'
	#   coordinate: 'camera' or 'lidar'
	# Output:
	#   (N, N', 7)
	boxes3d = []
	if cls == 'Car':
		acc_cls = ['Car', 'Van']
	elif cls == 'Pedestrian':
		acc_cls = ['Pedestrian']
	elif cls == 'Cyclist':
		acc_cls = ['Cyclist']
	else: # all
		acc_cls = []

	for label in labels:
		boxes3d_a_label = []
		for line in label:
			ret = line.split()
			if ret[0] in acc_cls or acc_cls == []:
				h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
				box3d = np.array([x, y, z, h, w, l, r])
				boxes3d_a_label.append(box3d)
		if coordinate == 'lidar':
			boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label))

		boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))
	return boxes3d


def box3d_to_label(batch_box3d, batch_cls, batch_score=[], coordinate='camera'):
	# Input:
	#   (N, N', 7) x y z h w l r
	#   (N, N')
	#   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
	#   coordinate(input): 'camera' or 'lidar'
	# Output:
	#   label: (N, N') N batches and N lines
	batch_label = []
	if batch_score:
		template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
		for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
			label = []
			for box, score, cls in zip(boxes, scores, clses):
				if coordinate == 'camera':
					box3d = box
					box2d = lidar_box3d_to_camera_box(
						camera_to_lidar_box(box[np.newaxis, :].astype(np.float32)), cal_projection=False)[0]
				else:
					box3d = lidar_to_camera_box(
						box[np.newaxis, :].astype(np.float32))[0]
					box2d = lidar_box3d_to_camera_box(
						box[np.newaxis, :].astype(np.float32), cal_projection=False)[0]
				x, y, z, h, w, l, r = box3d
				box3d = [h, w, l, x, y, z, r]
				label.append(template.format(
					cls, 0, 0, 0, *box2d, *box3d, float(score)))
			batch_label.append(label)
	else:
		template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
		for boxes, clses in zip(batch_box3d, batch_cls):
			label = []
			for box, cls in zip(boxes, clses):
				if coordinate == 'camera':
					box3d = box
					box2d = lidar_box3d_to_camera_box(
						camera_to_lidar_box(box[np.newaxis, :].astype(np.float32)), cal_projection=False)[0]
				else:
					box3d = lidar_to_camera_box(
						box[np.newaxis, :].astype(np.float32))[0]
					box2d = lidar_box3d_to_camera_box(
						box[np.newaxis, :].astype(np.float32), cal_projection=False)[0]
				x, y, z, h, w, l, r = box3d
				box3d = [h, w, l, x, y, z, r]
				label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
			batch_label.append(label)

	return np.array(batch_label)


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
	# Input:
	#   points: (N, 3)
	#   rx/y/z: in radians
	# Output:
	#   points: (N, 3)
	N = points.shape[0]
	points = np.hstack([points, np.ones((N, 1))])

	mat1 = np.eye(4)
	mat1[3, 0:3] = tx, ty, tz
	points = np.matmul(points, mat1)

	if rx != 0:
		mat = np.zeros((4, 4))
		mat[0, 0] = 1
		mat[3, 3] = 1
		mat[1, 1] = np.cos(rx)
		mat[1, 2] = -np.sin(rx)
		mat[2, 1] = np.sin(rx)
		mat[2, 2] = np.cos(rx)
		points = np.matmul(points, mat)

	if ry != 0:
		mat = np.zeros((4, 4))
		mat[1, 1] = 1
		mat[3, 3] = 1
		mat[0, 0] = np.cos(ry)
		mat[0, 2] = np.sin(ry)
		mat[2, 0] = -np.sin(ry)
		mat[2, 2] = np.cos(ry)
		points = np.matmul(points, mat)

	if rz != 0:
		mat = np.zeros((4, 4))
		mat[2, 2] = 1
		mat[3, 3] = 1
		mat[0, 0] = np.cos(rz)
		mat[0, 1] = -np.sin(rz)
		mat[1, 0] = np.sin(rz)
		mat[1, 1] = np.cos(rz)
		points = np.matmul(points, mat)

	return points[:, 0:3]


def box_transform(boxes, tx, ty, tz, r=0, coordinate='lidar'):
	# Input:
	#   boxes: (N, 7) x y z h w l rz/y
	# Output:
	#   boxes: (N, 7) x y z h w l rz/y
	boxes_corner = center_to_corner_box3d(
		boxes, coordinate=coordinate)  # (N, 8, 3)
	for idx in range(len(boxes_corner)):
		if coordinate == 'lidar':
			boxes_corner[idx] = point_transform(
				boxes_corner[idx], tx, ty, tz, rz=r)
		else:
			boxes_corner[idx] = point_transform(
				boxes_corner[idx], tx, ty, tz, ry=r)

	return corner_to_center_box3d(boxes_corner, coordinate=coordinate)


def cal_iou2d(box1, box2):
	# Input: 
	#   box1/2: x, y, w, l, r
	# Output :
	#   iou
	buf1 = np.zeros((700, 800, 3))
	buf2 = np.zeros((700, 800, 3))
	tmp = center_to_corner_box2d(np.array([box1, box2]), coordinate='lidar')
	box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
	box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
	buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
	buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]
	indiv = np.sum(np.absolute(buf1-buf2))
	share = np.sum((buf1 + buf2) == 2)
	if indiv == 0:
		return 0.0 # when target is out of bound
	return share / (indiv + share)

def cal_z_intersect(cz1, h1, cz2, h2):
	b1z1, b1z2 = cz1 - h1 / 2, cz1 + h1 / 2
	b2z1, b2z2 = cz2 - h2 / 2, cz2 + h2 / 2
	if b1z1 > b2z2 or b2z1 > b1z2:
		return 0
	elif b2z1 <= b1z1 <= b2z2:
		if b1z2 <= b2z2:
			return h1 / h2
		else:
			return (b2z2 - b1z1) / (b1z2 - b2z1)
	elif b1z1 < b2z1 < b1z2:
		if b2z2 <= b1z2:
			return h2 / h1
		else:
			return (b1z2 - b2z1) / (b2z2 - b1z1)


def cal_iou3d(box1, box2):
	# Input:
	#   box1/2: x, y, z, h, w, l, r
	# Output:
	#   iou
	buf1 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))
	buf2 = np.zeros((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 3))
	tmp = center_to_corner_box2d(np.array([box1[[0,1,4,5,6]], box2[[0,1,4,5,6]]]), coordinate='lidar')
	box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
	box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
	buf1 = cv2.fillConvexPoly(buf1, box1_corner, color=(1,1,1))[..., 0]
	buf2 = cv2.fillConvexPoly(buf2, box2_corner, color=(1,1,1))[..., 0]
	share = np.sum((buf1 + buf2) == 2)
	area1 = np.sum(buf1)
	area2 = np.sum(buf2)
	
	z1, h1, z2, h2 = box1[2], box1[3], box2[2], box2[3]
	z_intersect = cal_z_intersect(z1, h1, z2, h2)

	return share * z_intersect / (area1 * h1 + area2 * h2 - share * z_intersect)


def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d=0):
	# Inputs:
	#   boxes3d: (N1, 7) x,y,z,h,w,l,r
	#   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
	# Outputs:
	#   iou: (N1, N2)
	N1 = len(boxes3d)
	N2 = len(gt_boxes3d)
	output = np.zeros((N1, N2), dtype=np.float32)

	for idx in range(N1):
		for idy in range(N2):
			if cal_3d:
				output[idx, idy] = float(
					cal_iou3d(boxes3d[idx], gt_boxes3d[idy]))
			else:
				output[idx, idy] = float(
					cal_iou2d(boxes3d[idx, [0, 1, 4, 5, 6]], gt_boxes3d[idy, [0, 1, 4, 5, 6]]))

	return output


def cal_box2d_iou(boxes2d, gt_boxes2d):
	# Inputs:
	#   boxes2d: (N1, 5) x,y,w,l,r
	#   gt_boxes2d: (N2, 5) x,y,w,l,r
	# Outputs:
	#   iou: (N1, N2)
	N1 = len(boxes2d)
	N2 = len(gt_boxes2d)
	output = np.zeros((N1, N2), dtype=np.float32)
	for idx in range(N1):
		for idy in range(N2):
			output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])

	return output


def voxelNetAugScheme(lidar, labels, augData):
	np.random.seed()
	
	gt_box3d = labels  # (N', 7) x, y, z, h, w, l, r; camera coordinates

	'''
	Randomly choose between 0-3, equal probability
	0: Perturbation
	1: Rotation
	2: Scaling
	3: No augmentation
	'''
	choice = np.random.randint(low=0, high=4)

	if augData and choice == 0:
		# perturbation
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)
		lidar_corner_gt_box3d = center_to_corner_box3d(
			lidar_center_gt_box3d, coordinate='lidar')

		for idx in range(len(lidar_corner_gt_box3d)):
			# TODO: precisely gather the point
			is_collision = True
			_count = 0
			while is_collision and _count < 100:
				t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
				t_x = np.random.normal()
				t_y = np.random.normal()
				t_z = np.random.normal()
				# check collision
				tmp = box_transform(
					lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
				is_collision = False
				for idy in range(idx):
					x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
					x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
						0, 1, 4, 5, 6]]
					iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
									np.array([x2, y2, w2, l2, r2], dtype=np.float32))
					if iou > 0:
						is_collision = True
						_count += 1
						break
			if not is_collision:
				box_corner = lidar_corner_gt_box3d[idx]
				minx = np.min(box_corner[:, 0])
				miny = np.min(box_corner[:, 1])
				minz = np.min(box_corner[:, 2])
				maxx = np.max(box_corner[:, 0])
				maxy = np.max(box_corner[:, 1])
				maxz = np.max(box_corner[:, 2])
				bound_x = np.logical_and(
					lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
				bound_y = np.logical_and(
					lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
				bound_z = np.logical_and(
					lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
				bound_box = np.logical_and(
					np.logical_and(bound_x, bound_y), bound_z)
				lidar[bound_box, 0:3] = point_transform(
					lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
				lidar_center_gt_box3d[idx] = box_transform(
					lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')

	elif augData and choice == 1:
		# global rotation
		angle = np.random.uniform(-np.pi / 4, np.pi / 4)
		lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)
		lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
	
	elif augData and choice == 2:
		# global scaling
		factor = np.random.uniform(0.95, 1.05)
		lidar[:, 0:3] = lidar[:, 0:3] * factor
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)
		lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
	else:
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)

	return lidar, lidar_center_gt_box3d

def pixorAugScheme(lidar, labels, augData):	
	gt_box3d = labels  # (N', 7) x, y, z, h, w, l, r; camera coordinates

	'''
	Randomly choose between 0-2, equal probability
	0: Rotation
	1: flipping about X axis
	2: No augmentation
	'''
	choice = np.random.randint(low=0, high=3)

	if augData and choice == 0:
		# rotation
		# global rotation
		angle = np.random.uniform(-np.pi / 36, np.pi / 36)
		lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)
		lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')

	elif augData and choice == 1:
		# random flip
		lidar[:, 1] = -lidar[:, 1]
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)
		lidar_center_gt_box3d[:, 1] = -lidar_center_gt_box3d[:, 1]

	else:
		# no augmentation
		lidar_center_gt_box3d = camera_to_lidar_box_1(gt_box3d)

	return lidar, lidar_center_gt_box3d