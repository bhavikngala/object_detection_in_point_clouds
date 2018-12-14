import torch

import numpy as np
import math
import cv2

def lidarToBEV(lidar, gridConfig):
    '''
    Converts lidar data to Bird's Eye View as defined in PIXOR paper
    Arguments:
        lidar: LiDAR data - x, y, z, reflectance
        gridConfig: physical dimension range of the Region of interest
        and resolution of the grid
    '''

    # ranges of length, width, height; and resolution value
    x_r, y_r, z_r = gridConfig['x'], gridConfig['y'], gridConfig['z']
    res = gridConfig['res']

    bev = np.zeros((int((z_r[1]-z_r[0])/res + 1), int((y_r[1]-y_r[0])/res), int((x_r[1]-x_r[0])/res)), dtype='float32')

    mask = (lidar[:,0]>x_r[0]) & (lidar[:,0]<x_r[1]) & (lidar[:,1]>y_r[0]) & (lidar[:,1]<y_r[1]) & (lidar[:,2]>z_r[0]) & (lidar[:,2]<z_r[1])
    indices = lidar[mask][:,:3]/res
    ref = lidar[mask][:,3]/255.0

    indices = indices.astype(int)

    # axis rotation and origin shift
    # x = -y - int(y_r[0]/res)
    # y = x
    # z = -z + int(z_r[1]/res)

    bev[-indices[:,2]+int(z_r[1]/res), -indices[:, 1]-int(y_r[0]/res), indices[:, 0]] = 1
    bev[-1, -indices[:, 1]-int(y_r[0]/res), indices[:, 0]] = ref

    return bev


class TargetParameterization():


    def __init__(self, gridConfig, gridL, gridW, downSamplingFactor=4, device=None):
        self.xRange = gridConfig['x']
        self.yRange = gridConfig['y']
        self.zRange = gridConfig['z']
        self.res = gridConfig['res']
        self.outputGridRes = self.res*downSamplingFactor
        self.gridL = gridL
        self.gridW = gridW
        self.device = device

        self.yy, self.xx = torch.meshgrid(
                [torch.arange(self.yRange[0], self.yRange[1], self.outputGridRes, dtype=torch.float32, device=device),
                 torch.arange(self.xRange[1], self.xRange[0], -self.outputGridRes, dtype=torch.float32, device=device)])
        # self.yy = self.yy - self.yRange[0]


    def encodeLabelToYolo(self, labels):
        # just yolo, only the cell containing the centre is responsible for it
        # labels -> c, cx, cy, cz, H, W, L, r
        r, c = self.xx.size()
        targetClass = torch.zeros((r, c), dtype=torch.float32, device=self.device)
        targetLoc = torch.zeros((r, c, 6), dtype=torch.float32, device=self.device)

        for i in range(labels.shape[0]):
            c, cx, cy, cz, H, W, L, ry = labels[i,:]

            mask = (cx <= self.xx) & (cx > (self.xx - self.outputGridRes)) & \
                   (cy >= self.yy) & (cy < (self.yy + self.outputGridRes))

            if mask.sum()==1:
                gridX = self.xx[mask]
                gridY = self.yy[mask]
                t = torch.tensor([torch.cos(2*ry), torch.sin(2*ry), \
                                  gridX - cx, cy - gridY, \
                                  torch.log(L/self.gridL), \
                                  torch.log(W/self.gridW)], dtype=torch.float32)
                targetLoc[mask] = t
                targetClass[mask] = 1.0

        return targetClass, targetLoc


    def encodeLabelToPIXORIgnoreBoundaryPix(self, labels, mean=None, std=None):
        # pixor style label encoding, all pixels inside ground truth
        # box are positive samples rest are negative
        # labels -> c, cx, cy, cz, H, W, L, r
        r, c = self.xx.size()
        # targetClass = torch.zeros((r, c), dtype=torch.float32, device=self.device)
        # targetLoc = torch.zeros((r, c, 6), dtype=torch.float32, device=self.device)
        targetClass = np.zeros((r, c), dtype=np.float32)
        targetLoc = np.zeros((r, c, 6), dtype=np.float32)

        for i in range(labels.shape[0]):
            cl, cx, cy, cz, H, W, L, ry = labels[i,:]

            L03, W03 = 0.3 * L.item(), 0.3 * W.item()
            L12, W12 = 1.2 * L.item(), 1.2 * W.item()

            gt03 = np.array([[L03/2,  L03/2, -L03/2, -L03/2],
                             [W03/2, -W03/2, -W03/2,  W03/2],
                             [    0,      0,      0,      0]], dtype=np.float32)
            gt12 = np.array([[L12/2,  L12/2, -L12/2, -L12/2],
                             [W12/2, -W12/2, -W12/2,  W12/2],
                             [    0,      0,      0,      0]], dtype=np.float32)
            gt03 = cart2hom(gt03.T)
            gt12 = cart2hom(gt12.T)

            R = rotz(ry)
            translation = np.array([cx.item(), cy.item(), 0], dtype=np.float32)
            transformationMatrix = transform_from_rot_trans(R, translation)

            gt03 = (np.matmul(transformationMatrix, gt03.T)).T[:,[0,1]]
            gt12 = (np.matmul(transformationMatrix, gt12.T)).T[:,[0,1]]
            gt03 = gt03.astype(np.int32)
            gt12 = gt12.astype(np.int32)

            targetClass = cv2.fillConvexPoly(targetClass, gt12, -1)
            # targetClass = cv2.fillConvexPoly(targetClass, gt03, 1)
            
            # convert velo to matrix indices
            # r' = r - (x_velo - x_grid_min)/gridRes
            # c' = (y_velo - y_grid_min)/gridRes
            veloToMat = gt03.copy()
            veloToMat[:,0] = veloToMat[:,0]-self.xRange[0]
            veloToMat[:,0] = veloToMat[:,0]/self.outputGridRes
            veloToMat[:,0] = c - veloToMat[:,0]
            
            veloToMat[:,1] = veloToMat[:,1]-self.yRange[0]
            veloToMat[:,1] = veloToMat[:,1]/self.outputGridRes
            rmin, cmin = veloToMat.min(axis=0)
            rmax, cmax = veloToMat.max(axis=0)
            for rprime in range(rmin-1, rmax, 1):
                for cprime in range(cmin-1, cmax, 1):
                    if cv2.pointPolygonTest(veloToMat, (rprime, cprime), False) >= 0:
                        t = torch.tensor([torch.cos(2*ry), torch.sin(2*ry), \
                                          self.xx[cprime,rprime] - cx, \
                                          cy - self.yy[cprime,rprime], \
                                          torch.log(L/self.gridL), \
                                          torch.log(W/self.gridW)], dtype=torch.float32)
                        if mean is not None and std is not None:
                            t = (t-mean)/std
                        # print('cx:',cx.item(),'xx:',self.xx[cprime,rprime].item(), 'L03',L03,'cy:',cy.item(),'yy:',self.yy[cprime,rprime].item(), 'W03',W03)
                        targetLoc[cprime, rprime] = t
                        targetClass[cprime, rprime] = 1.0

        return torch.from_numpy(targetClass), torch.from_numpy(targetLoc)
        

    def decodeYoloToLabel(self, networkOutput):
        networkOutput[:,:,0] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])/2
        networkOutput[:,:,1] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])/2
        networkOutput[:,:,2] = networkOutput[:,:,2] + self.xx
        networkOutput[:,:,3] = networkOutput[:,:,3] + self.yy + self.yRange[0]
        networkOutput[:,:,4] = torch.exp(networkOutput[:,:,4]) * self.gridL
        networkOutput[:,:,5] = torch.exp(networkOutput[:,:,5]) * self.gridW

        return networkOutput


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


def cart2hom(pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom