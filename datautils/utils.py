import torch

import numpy as np
import math
import config as cnf

def lidarToBEV(lidar, gridConfig=cnf.gridConfig):
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

        self.yy, self.xx = torch.meshdrig(
                [torch.arange(self.yRange[0], self.yRange[1], self.outputGridRes, dtype=torch.float32, device=device),
                 torch.arange(self.xRange[1], self.xRange[0], -self.outputGridRes, dtype=torch.float32, device=device)])
        self.yy = self.yy - self.yRange[0]


    def encodeLabelToYolo(labels):
        r, c = self.xx.size()
        targetClass = torch.zeros((r, c), dtype=torch.float32, device=self.device)
        targetLoc = torch.zeros((r, c, 6), dtype=torch.float32, device=self.device)

        for i in range(labels.size(0)):
            c, cx, cy, cz, H, W, L, r = labels[i,:]

            mask = (cx <= self.xx) & (cx > (self.xx - self.outputGridRes)) & \
                   (cy >= self.yy) & (cy < (self.yy + self.outputGridRes))
            
            if mask.sum()==1:
                gridX = self.xx[mask]
                gridY = self.yy[mask]

                targetLoc[mask][0] = torch.cos(2*r)
                targetLoc[mask][1] = torch.sin(2*r)
                targetLoc[mask][2] = gridX - cx
                targetLoc[mask][3] = cy - gridY
                targetLoc[mask][4] = torch.log(L/self.gridL)
                targetLos[mask][5] = torch.log(W/self.gridW)
                targetClass[mask] = 1.0

        if targetClass.sum() > 0:
            return targetClass, targetLoc
        else:
            return targetClass, None

        

    def decodeYoloToLabel(networkOutput):
        networkOutput[:,:,0] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])/2
        networkOutput[:,:,1] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])/2
        networkOutput[:,:,2] = networkOutput[:,:,2] + self.xx
        networkOutput[:,:,3] = networkOutput[:,:,3] + self.yy + self.yRange[0]
        networkOutput[:,:,4] = torch.exp(networkOutput[:,:,4]) * self.gridL
        networkOutput[:,:,5] = torch.exp(networkOutput[:,:,5]) * self.gridW

        return networkOutput


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    filename = './../data/tiny_set/train/000492.bin'
    gridConfig = {
        'x':(0, 70.4),
        'y':(-40, 40),
        'z':(-2.5, 1),
        'res':0.1
    }
    lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    bev = lidar_to_top(lidar, gridConfig)
    print(bev.shape)
    plt.imshow(bev[-1,:,:])
    plt.show()