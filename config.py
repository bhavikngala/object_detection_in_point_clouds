rootDir = './../data/KITTI_BEV'
batchSize = 35

gridConfig = {
	'x':(0, 20),
	'y':(-20, 20),
	'z':(-2.5, 1),
	'res':0.1
}

objtype = 'Car'

# res_block_layers = list if number of channels in the first conv layer of each res_block
# up_sample_layers = list of tuple of number of channels input deconv and conv layers

res_block_layers = [24, 48, 64, 96]
up_sample_layers = [(196, 256), (128, 192)]

# training parameters
learningRate = 1e-4
epochs = 100