import numpy as np
import matplotlib.pyplot as plt
import datautils.kittiUtils as ku

def lidarToBEV(lidar, gridConfig):
    '''
    Converts lidar data to Bird's Eye View as defined in PIXOR paper
    Arguments:
        lidar: LiDAR data - x, y, z, reflectance
        gridConfig: physical dimension range of the Region of interest
        and resolution of the grid
    '''
    # shape of lidar array should be (-1, 4)
    assert(lidar.shape[1] == 4)

    # x, y, z, reflectance values
    x = lidar[:, 0]
    y = lidar[:, 1]
    z = lidar[:, 2]
    r = lidar[:, 3]

    # ranges of length, width, height; and resolution value
    x_r, y_r, z_r = gridConfig['x'], gridConfig['y'], gridConfig['z']
    res = gridConfig['res']    

    # bev tensor
    bev = np.zeros((int((z_r[1]-z_r[0])/res + 1), int((y_r[1]-y_r[0])/res), int((x_r[1]-x_r[0])/res)), dtype='float32')
    bev1 = np.zeros((int((y_r[1]-y_r[0])/res), int((x_r[1]-x_r[0])/res)), dtype='float32')

    for i in range(lidar.shape[0]):
        if (x[i]>x_r[0] and x[i]<x_r[1]) and (y[i]>y_r[0] and y[i]<y_r[1]) and (z[i]>z_r[0] and z[i]<z_r[1]):
            x_index = int(-y[i]/res)
            y_index = int(x[i]/res)
            z_index = int(-z[i]/res)

            # shifting to new origin
            x_index -= int(y_r[0]/res)
            z_index += int(z_r[1]/res)

            bev[z_index, x_index, y_index] = 1
            #  normalize the reflectance value
            bev1[x_index, y_index] = r[i]/255.0    

    bev[-1, :, :] = bev1
    
    # plt.imshow(bev[:,:,35])
    # plt.show()

    return bev

def rotateFrame(lidar, targets):
    '''
    Requires: lidar - numpy array, shape (-1,4); x, y, z, reflectance
    Requires: targets - numpy array of labels - [x, y, z, h, w, l, ry]
    Returns : lidar - rotated by random theta
    Returns : targets - numpy array of shape (-1, 7) [x, y, z, h, w, l, ry]
    '''
    # random rotating angle
    theta = np.random.uniform(low=-np.pi/4, high=(np.pi/4+np.pi/180))
    # transformation matrix
    tmat = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]], dtype='float32')
    # transform lidar data
    lidar[:, :3] = np.matmul(tmat, lidar[:, :3].T).T

    # transform labels from camera system to lidar system
    targets = ku.camera_to_lidar_box(targets.tolist())

    # transfrom the label points
    targets[:, :3] = np.matmul(tmat, targets[:, :3].T).T
    targets[:, :6] = targets[:, :6] - theta

    return lidar, targets

def scaleFrame(lidar, targets):
    # random scaling sample
    scale = np.random.uniform(low=0.95, high=1.06)
    # transformation matrix
    tmat = np.array([[scale,     0,     0],
                     [    0, scale,     0],
                     [    0,     0, scale]], dtype='float32')

    # transform lidar data
    lidar[:,:3] = np.matmul(tmat, lidar[:, :3].T).T

    # transform labels from camera system to lidar system
    targets = ku.camera_to_lidar_box(targets.tolist())

    # transfrom the label points
    targets[:, :3] = np.matmul(tmat, targets[:, :3].T).T

    return lidar, targets

def perturbFrame(lidar, targets):
    # apply perturbations to bounding boxes as described in the paper
    # VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
    # https://arxiv.org/abs/1711.06396
    return lidar, targets