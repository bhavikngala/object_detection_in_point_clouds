import numpy as np
import math
import config as cnf
'''
def lidarToBEV(lidar, gridConfig=cnf.gridConfig):
    \'''
    Converts lidar data to Bird's Eye View as defined in PIXOR paper
    Arguments:
        lidar: LiDAR data - x, y, z, reflectance
        gridConfig: physical dimension range of the Region of interest
        and resolution of the grid
    \'''

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
'''
def lidarToBEV(lidar, gridConfig=cnf.gridConfig):
    x_r, y_r, z_r = gridConfig['x'], gridConfig['y'], gridConfig['z']
    res = gridConfig['res']

    idx = np.where (lidar[:,0]>x_r[0])
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<x_r[1])
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>y_r[0])
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<y_r[1])
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>z_r[0])
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<z_r[1])
    lidar = lidar[idx]



    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    
    qxs=((pxs-x_r[0])//res).astype(np.int32)
    qys=((pys-y_r[0])//res).astype(np.int32)
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=((pzs-z_r[0])/res).astype(np.int32)
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()
    # print(quantized.shape)
    
    X0, Xn = 0, int((x_r[1]-x_r[0])//res)
    Y0, Yn = 0, int((y_r[1]-y_r[0])//res)+1
    Z0, Zn = 0, int((z_r[1]-z_r[0])/res)
    # print(Xn)
    # print(Yn)
    # print(Zn)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 1
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    # top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    top = np.zeros(shape=(channel,width,height), dtype=np.float32)
    # print(top.shape)

    
    # print(quantized)
    # if 1:  #new method
    for x in range(Xn):
        ix  = np.where(quantized[:,0]==x)
        # print(ix)
        quantized_x = quantized[ix]
        if len(quantized_x) == 0 : continue
        yy = -x

        for y in range(Yn):
            iy  = np.where(quantized_x[:,1]==y)
            quantized_xy = quantized_x[iy]
            count = len(quantized_xy)
            if  count==0 : continue
            xx = -y

            top[Zn,xx,yy] = min(1, np.log(count+1)/math.log(32))
            max_height_point = np.argmax(quantized_xy[:,2])
            top[Zn,xx,yy]=quantized_xy[max_height_point, 3]
            
            for z in range(Zn):
                iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                quantized_xyz = quantized_xy[iz]
                if len(quantized_xyz) == 0 : continue
                zz = z

                #height per slice
                max_height = max(0,np.max(quantized_xyz[:,2])-z)
                # print('max ht is ',max_height)
                top[zz,xx,yy]=max_height
                # print(quantized_xyz)
    # top = top.permute(2,1,0)
    return top

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