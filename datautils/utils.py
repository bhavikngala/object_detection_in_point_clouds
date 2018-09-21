import numpy as np
import matplotlib.pyplot as plt

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
    bev = np.zeros((int((y_r[1]-y_r[0])/res), int((x_r[1]-x_r[0])/res), int((z_r[1]-z_r[0])/res + 1)), dtype='float32')
    bev1 = np.zeros((int((y_r[1]-y_r[0])/res), int((x_r[1]-x_r[0])/res)), dtype='float32')

    for i in range(lidar.shape[0]):
        if (x[i]>x_r[0] and x[i]<x_r[1]) and (y[i]>y_r[0] and y[i]<y_r[1]) and (z[i]>z_r[0] and z[i]<z_r[1]):
            x_index = int(-y[i]/res)
            y_index = int(x[i]/res)
            z_index = int(-z[i]/res)

            # shifting to new origin
            x_index -= int(y_r[0]/res)
            # y_index = 
            z_index += int(z_r[1]/res)

            bev[x_index, y_index, z_index] = 1
            bev1[x_index, y_index] = r[i]/255.0
            # if z[i] >= z_r[1]-res:
            #     bev[x_index, y_index, bev.shape[2]-1] = r[i]      

    bev[:, :, bev.shape[2]-1] = bev1
    
    # plt.imshow(bev[:,:,35])
    # plt.show()

    return bev