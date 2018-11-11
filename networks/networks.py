import torch
import torch.nn as nn
from networks.blocks import Bottleneck_3, Bottleneck_6, Upsample

# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class PointCloudDetector(nn.Module):

	# res_block_layers = list if number of channels in the first conv layer of each res_block
	# up_sample_layers = list of tuple of number of channels input deconv and conv layers
	# up_sample_deconv = tuplr of (dilation, stride, padding, output_padding) for deconvolution in upsampling
	def __init__(self, res_block_layers, up_sample_layers, up_sample_deconv):

		super(PointCloudDetector, self).__init__()

		# keep padding in mind since at all convolutions are "SAME"
		# except for layers where downsampling happens

		self.conv1 = nn.Conv2d(in_channels=36, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)

		self.res_block1 = Bottleneck_3(in_channels = 32, out_channels=res_block_layers[0])
		self.bn_res_block1 = nn.BatchNorm2d(4 * res_block_layers[0])

		self.res_block2 = Bottleneck_6(in_channels = 4 * res_block_layers[0], out_channels=res_block_layers[1])
		self.bn_res_block2 = nn.BatchNorm2d(4 * res_block_layers[1])

		self.res_block3 = Bottleneck_6(in_channels = 4 * res_block_layers[1], out_channels=res_block_layers[2])
		self.bn_res_block3 = nn.BatchNorm2d(4 * res_block_layers[2])

		self.res_block4 = Bottleneck_3(in_channels = 4 * res_block_layers[2], out_channels=res_block_layers[3])
		self.bn_res_block4 = nn.BatchNorm2d(4 * res_block_layers[3])
		
		self.conv4 = nn.Conv2d(in_channels=4*res_block_layers[3], out_channels=196, kernel_size=1, bias=False)
		self.bn4 = nn.BatchNorm2d(196)

		self.upsample1 = Upsample(in_channels = up_sample_layers[0], out_channels = 128, args = up_sample_deconv[0])
		self.bn_upsample1 = nn.BatchNorm2d(128)

		self.upsample2 = Upsample(in_channels = up_sample_layers[1], out_channels = 96, args = up_sample_deconv[1])
		self.bn_upsample2 = nn.BatchNorm2d(96)
		
		self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
		self.bn5 = nn.BatchNorm2d(96)
		
		self.conv6 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
		self.bn6 = nn.BatchNorm2d(96)
		
		self.conv7 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
		self.bn7 = nn.BatchNorm2d(96)
		
		self.conv8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
		self.bn8 = nn.BatchNorm2d(96)

		# sigmoid activation
		self.conv_cla = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=3, padding=1, bias=False)

		# check activation function from SSD code
		self.conv_loc = nn.Conv2d(in_channels=96, out_channels=6, kernel_size=3, padding=1, bias=False)

		self.relu = nn.ReLU(inplace=True)
		self.cla_act = nn.Sigmoid()
		self.loc_act = None

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.res_block1(x)
		x = self.bn_res_block1(x)
		x = self.relu(x)

		res_2 = self.res_block2(x)
		res_2 = self.bn_res_block2(res_2)
		res_2 = self.relu(res_2)

		res_3 = self.res_block3(res_2)
		res_3 = self.bn_res_block3(res_3)
		res_3 = self.relu(res_3)

		x = self.res_block4(res_3)
		x = self.bn_res_block4(x)
		x = self.relu(x)

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)

		x = self.upsample1(x, res_3)
		x = self.bn_upsample1(x)
		x = self.relu(x)

		x = self.upsample2(x, res_2)
		x = self.bn_upsample2(x)
		x = self.relu(x)

		x = self.conv5(x)
		x = self.bn5(x)
		x = self.relu(x)

		x = self.conv6(x)
		x = self.bn6(x)
		x = self.relu(x)

		x = self.conv7(x)
		x = self.bn7(x)
		x = self.relu(x)

		x = self.conv8(x)
		x = self.bn8(x)
		x = self.relu(x)

		cla = self.conv_cla(x)
		cla = self.cla_act(cla)

		loc = self.conv_loc(x)
		if self.loc_act:
			loc = self.loc_act(loc)

		return cla, loc