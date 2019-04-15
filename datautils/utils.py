import torch

import numpy as np
import math
import cv2
from shapely.geometry import Polygon

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


    def __init__(self, cordinate, gridConfig, gridL, gridW, downSamplingFactor=4, device=None):
        self.cordinate = cordinate
        if cordinate == 'velo':
            self.veloCordInit(gridConfig, gridL, gridW, downSamplingFactor, device)
        elif cordinate == 'cam':
            self.camCordInit(gridConfig, gridL, gridW, downSamplingFactor, device)
        

    def veloCordInit(self, gridConfig, gridL, gridW, downSamplingFactor=4, device=None):
        self.xRange = gridConfig['x']
        self.yRange = gridConfig['y']
        self.zRange = gridConfig['z']
        self.res = gridConfig['res']
        self.outputGridRes = self.res*downSamplingFactor
        self.gridL = gridL
        self.gridW = gridW
        self.device = device

        x = torch.arange(self.xRange[0], self.xRange[1], self.outputGridRes, dtype=torch.float32, device=device)
        if device is not None:
            x = x[:-1]
        y = torch.arange(self.yRange[1], self.yRange[0], -self.outputGridRes, dtype=torch.float32, device=device)
        self.yy, self.xx = torch.meshgrid([y, x])


    def camCordInit(self, gridConfig, gridL, gridW, downSamplingFactor=4, device=None):
        self.xRange = gridConfig['y']
        self.yRange = gridConfig['z']
        self.zRange = gridConfig['x']
        self.res = gridConfig['res']
        self.outputGridRes = self.res*downSamplingFactor
        self.gridL = gridL
        self.gridW = gridW
        self.device = device

        x = torch.arange(self.xRange[0], self.xRange[1], self.outputGridRes, dtype=torch.float32, device=device)
        if device is not None:
            x = x[:-1]
        z = torch.arange(self.yRange[0], self.yRange[1], self.outputGridRes, dtype=torch.float32, device=device)
        self.xx, self.yy = torch.meshgrid([x, z])


    def encodeLabel(self, labels, mean=None, std=None):
        if self.cordinate == 'velo':
            return self.encodeLabelToPIXORIgnoreBoundaryPixVeloCord(labels, mean, std)
        elif self.cordinate == 'cam':
            return self.encodeLabelToPIXORIgnoreBoundaryPixCamCord(labels, mean, std)


    def encodeLabelToPIXORIgnoreBoundaryPixCamCord(self, labels, mean=None, std=None):
        print('inside encodeLabelToPIXORIgnoreBoundaryPixCamCord')
        # labels -> c, cx, cy, cz, H, W, L, r
        r, c = self.xx.size()
        targetClass = np.zeros((r, c), dtype=np.float32)
        targetLoc = np.zeros((r, c, 6), dtype=np.float32)
        
        for i in range(labels.shape[0]):
            cl, cx, cy, cz, H, W, L, ry = labels[i,:]

            L03, W03, H03 = 0.3 * L.item(), 0.3 * W.item(), 0.3 * H.item()
            L12, W12, H12 = 1.2 * L.item(), 1.2 * W.item(), 1.2 * H.item()
            
            gt03 = np.array([[ W03/2,  L03/2,  H03/2],
                             [-W03/2,  L03/2,  H03/2],
                             [-W03/2, -L03/2,  H03/2],
                             [ W03/2, -L03/2,  H03/2],
                             [ W03/2,  L03/2, -H03/2],
                             [-W03/2,  L03/2, -H03/2],
                             [-W03/2, -L03/2, -H03/2],
                             [ W03/2, -L03/2, -H03/2]], dtype=np.float32)
            gt12 = np.array([[ W12/2,  L12/2,  H12/2],
                             [-W12/2,  L12/2,  H12/2],
                             [-W12/2, -L12/2,  H12/2],
                             [ W12/2, -L12/2,  H12/2],
                             [ W12/2,  L12/2, -H12/2],
                             [-W12/2,  L12/2, -H12/2],
                             [-W12/2, -L12/2, -H12/2],
                             [ W12/2, -L12/2, -H12/2]], dtype=np.float32)
            gt03 = cart2hom(gt03)
            gt12 = cart2hom(gt12)

            R = rotz(-ry)
            translation = np.array([cx.item(), cy.item(), cz.item()], dtype=np.float32)
            transformationMatrix = transform_from_rot_trans(R, translation)

            gt03 = (np.matmul(transformationMatrix, gt03.T)).T[:4,[0,1]]
            gt12 = (np.matmul(transformationMatrix, gt12.T)).T[:4,[0,1]]
            # print('cx',cx.item(),'cy',cy.item(),'L',L.item(),'L12',L12,'W12','W',W.item(),W12)
            # print(gt12)
            gt03 = self.veloCordToMatrixIndices(gt03)
            gt12 = self.veloCordToMatrixIndices(gt12)
            # print('\nind\n',gt12)
            targetClass = cv2.fillConvexPoly(targetClass, gt12, -1)
            # targetClass = cv2.fillConvexPoly(targetClass, gt03, 1)
            
            cmin, rmin = gt03.min(axis=0)
            cmax, rmax = gt03.max(axis=0)
            for rprime in range(rmin, rmax+1, 1):
                for cprime in range(cmin, cmax+1, 1):
                    if cv2.pointPolygonTest(gt03, (cprime, rprime), False) >= 0:
                        # print('cx', cx.item(), 'xx', self.xx[rprime,cprime].item(),'cy', cy.item(),'yy', self.yy[rprime,cprime].item())
                        t = torch.tensor([torch.cos(ry), torch.sin(ry), \
                                          cx - self.xx[rprime,cprime], \
                                          cy - self.yy[rprime,cprime], \
                                          torch.log(L), \
                                          torch.log(W)], dtype=torch.float32)
                        if mean is not None and std is not None:
                            t = (t-mean)/std
                        targetLoc[rprime, cprime] = t
                        targetClass[rprime, cprime] = 1.0
            
        return torch.from_numpy(targetClass), torch.from_numpy(targetLoc)


    def camCordToMatrixIndices(self, cam):
        r, c = self.xx.size()
        cord = cam.copy()

        # x -> c'; cam_x = outputGridRes * r' + xRange[0]
        cord[:,0] = (cam[:,0]-self.xRange[0])/self.outputGridRes
        # y -> r'; velo_y = -outputGridRes * r' + yRange[1]
        cord[:,1] = (cam[:,1]-self.zRange[0])/self.outputGridRes

        cord[cord[:,0]>=c,0] = c-1
        cord[cord[:,0]<0,0] = 0
        cord[cord[:,1]>=r,1] = r-1
        cord[cord[:,1]<0,1] = 0
        cord = np.floor(cord)
        cord = cord.astype(np.int)
        return cord


    def encodeLabelToPIXORIgnoreBoundaryPixVeloCord(self, labels, mean=None, std=None):
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
            # print('------------')
            # print(cl.item(), cx.item(), cy.item(), cz.item(), H.item(), W.item(), L.item(), ry.item())
            # print('------------')

            L03, W03, H03 = 0.3 * L.item(), 0.3 * W.item(), 0.3 * H.item()
            L12, W12, H12 = 1.2 * L.item(), 1.2 * W.item(), 1.2 * H.item()
            
            gt03 = np.array([[ W03/2,  L03/2,  H03/2],
                             [ W03/2, -L03/2,  H03/2],
                             [-W03/2, -L03/2,  H03/2],
                             [-W03/2,  L03/2,  H03/2],
                             [ W03/2,  L03/2, -H03/2],
                             [ W03/2, -L03/2, -H03/2],
                             [-W03/2, -L03/2, -H03/2],
                             [-W03/2,  L03/2, -H03/2]], dtype=np.float32)
            gt12 = np.array([[ W12/2,  L12/2,  H12/2],
                             [ W12/2, -L12/2,  H12/2],
                             [-W12/2, -L12/2,  H12/2],
                             [-W12/2,  L12/2,  H12/2],
                             [ W12/2,  L12/2, -H12/2],
                             [ W12/2, -L12/2, -H12/2],
                             [-W12/2, -L12/2, -H12/2],
                             [-W12/2,  L12/2, -H12/2]], dtype=np.float32)
            gt03 = cart2hom(gt03)
            gt12 = cart2hom(gt12)

            R = rotz(-ry)
            translation = np.array([cx.item(), cy.item(), cz.item()], dtype=np.float32)
            transformationMatrix = transform_from_rot_trans(R, translation)

            gt03 = (np.matmul(transformationMatrix, gt03.T)).T[:4,[0,1]]
            gt12 = (np.matmul(transformationMatrix, gt12.T)).T[:4,[0,1]]
            # print('cx',cx.item(),'cy',cy.item(),'L',L.item(),'L12',L12,'W12','W',W.item(),W12)
            # print(gt12)
            gt03 = self.veloCordToMatrixIndices(gt03)
            gt12 = self.veloCordToMatrixIndices(gt12)
            # print('\nind\n',gt12)
            targetClass = cv2.fillConvexPoly(targetClass, gt12, -1)
            # targetClass = cv2.fillConvexPoly(targetClass, gt03, 1)
            
            cmin, rmin = gt03.min(axis=0)
            cmax, rmax = gt03.max(axis=0)
            # print('-----')
            for rprime in range(rmin, rmax+1, 1):
                for cprime in range(cmin, cmax+1, 1):
                    if cv2.pointPolygonTest(gt03, (cprime, rprime), False) >= 0:
                        # print('cx', cx.item(), 'xx', self.xx[rprime,cprime].item(),'cy', cy.item(),'yy', self.yy[rprime,cprime].item())
                        t = torch.tensor([torch.cos(ry), torch.sin(ry), \
                                          cx - self.xx[rprime,cprime], \
                                          cy - self.yy[rprime,cprime], \
                                          torch.log(L), \
                                          torch.log(W)], dtype=torch.float32)
                        # print(t.size(), mean.size(), std.size())
                        if mean is not None and std is not None:
                            t = (t-mean)/std
                        targetLoc[rprime, cprime] = t
                        targetClass[rprime, cprime] = 1.0
            
        return torch.from_numpy(targetClass), torch.from_numpy(targetLoc)


    def decodePIXORToLabel(self, networkOutput, mean=None, std=None):
        # print(networkOutput.size(), mean.size(), std.size())
        if mean is not None and std is not None:
            networkOutput = (networkOutput * std) + mean
        networkOutput[:,:,0] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])
        # networkOutput[:,:,1] = torch.atan2(networkOutput[:,:,1],networkOutput[:,:,0])
        networkOutput[:,:,2] = networkOutput[:,:,2] + self.xx
        networkOutput[:,:,3] = networkOutput[:,:,3] + self.yy
        networkOutput[:,:,4] = torch.exp(networkOutput[:,:,4])
        networkOutput[:,:,5] = torch.exp(networkOutput[:,:,5])
        return networkOutput


    def veloCordToMatrixIndices(self, velo):
        r, c = self.xx.size()
        cord = velo.copy()

        # x -> c'; velo_x = -outputGridRes * r' + xRange[0]
        cord[:,0] = (velo[:,0]-self.xRange[0])/self.outputGridRes
        # y -> r'; velo_y = -outputGridRes * r' + yRange[1]
        cord[:,1] = (self.yRange[1]-velo[:,1])/self.outputGridRes

        cord[cord[:,0]>=c,0] = c-1
        cord[cord[:,0]<0,0] = 0
        cord[cord[:,1]>=r,1] = r-1
        cord[cord[:,1]<0,1] = 0
        cord = np.floor(cord)
        cord = cord.astype(np.int)
        return cord


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


def center2BoxCorners(boxCenter):
    # center -> Nx7; [x, y, z, h, w, l, r]
    # output -> Nx8x3
    boxCorners = np.zeros((boxCenter.shape[0], 8, 3))
    for i in range(boxCenter.shape[0]):
        cx, cy, cz, H, W, L, ry = boxCenter[i]

        bc = np.array([[ L/2, -W/2,  H/2],
                       [ L/2,  W/2,  H/2],
                       [-L/2,  W/2,  H/2],
                       [-L/2, -W/2,  H/2],
                       [ L/2, -W/2, -H/2],
                       [ L/2,  W/2, -H/2],
                       [-L/2,  W/2, -H/2],
                       [-L/2, -W/2, -H/2]], dtype=np.float32)
        bc = cart2hom(bc)

        R = rotz(ry)
        translation = np.array([cx, cy, cz], dtype=np.float32)
        transformationMatrix = transform_from_rot_trans(R, translation)

        bc = (np.matmul(transformationMatrix, bc.T)).T
        
        boxCorners[i,:,:] = bc[:,:3]

    return boxCorners


def nmsPredictions(predL, predC, iouThreshold):
    # predL and predC are already thresholded with confidence > some value
    # return predL, predC

    sortIndices = np.argsort(predC)
    predC = predC[sortIndices]
    predL = predL[sortIndices]

    nmsPredL = []
    nmsPredC = []

    deletedIndices = set()

    for i in range(predL.shape[0]-1):
        if i in deletedIndices:
            continue

        theta, _, cx, cy, L, W = predL[i]

        boxCorners = center2BoxCorners(np.array([[cx, cy, 0, 0, W, L, theta]]))

        boxCorners = boxCorners[:,:4,:2].squeeze()

        polygon1 = Polygon([(boxCorners[0,0], boxCorners[0,1]),
                            (boxCorners[1,0], boxCorners[1,1]),
                            (boxCorners[2,0], boxCorners[2,1]),
                            (boxCorners[3,0], boxCorners[3,1])])

        for j in range(i+1, predL.shape[0]):
            if j in deletedIndices:
                continue

            theta, _, cx, cy, L, W = predL[j]

            boxCorners = center2BoxCorners(np.array([[cx, cy, 0, 0, W, L, theta]]))

            boxCorners = boxCorners[:,:4,:2].squeeze()

            polygon2 = Polygon([(boxCorners[0,0], boxCorners[0,1]),
                                (boxCorners[1,0], boxCorners[1,1]),
                                (boxCorners[2,0], boxCorners[2,1]),
                                (boxCorners[3,0], boxCorners[3,1])])

            iou = polygon1.intersection(polygon2).area/polygon1.union(polygon2).area

            if iou >= iouThreshold:
                deletedIndices.add(j)

        nmsPredL.append(predL[i])
        nmsPredC.append(predC[i])

    return np.array(nmsPredL), np.array(nmsPredC)


class AugmentLidar():
    '''
    # lidar data format : x, y, z, r
    # labels format     :
    '''

    def __init__(self, thetaRange):
        self.thetaRangeXAxis = thetaRange[0]
        self.thetaRangeYAxis = thetaRange[1]
        self.thetaRangeZAxis = thetaRange[2]


    def rotateLidarFrame(self, lidar, axis='z'):
        if axis=='z':
            theta = np.random.uniform(low = self.thetaRangeZAxis[0], high=self.thetaRangeZAxis[1])

        return 

    def rotateLabels(self, labels, axis='z'):
        raise NotImplementedError()


    def flipLidarFrame(self, lidar, axis='x'):
        raise NotImplementedError()


    def flipLabels(self, labels, axis='x'):
        raise NotImplementedError()


    def getAugmentationMethodsList(self):
        lidarAugmentationMethodsList = [self.rotateLidarFrame, self.flipLidarFrame]
        labelsAugmentationMethodsList = [self.rotateLabels, self.flipLabels]
        return (lidarAugmentationMethodsList, labelsAugmentationMethodsList)


class Augmentor(AugmentLidar):
    def __init__(self, thetaRange):
        super().__init__(thetaRange)

        self.lidarAugmentationMethodsList, self.labelsAugmentationMethodsList = \
            self.getAugmentationMethodsList()
        self.numAugmentationMethods = len(self.lidarAugmentationMethodsList)


    def augment(self, lidar, labels=None):
        augmentMethodNumber = np.random.randint(low=0, high=self.numAugmentationMethods)

        lidar = self.lidarAugmentationMethodsList[augmentMethodNumber](lidar)
        if labels:
            labels = self.labelsAugmentationMethodsList[augmentMethodNumber](labels)

        return lidar, labels
