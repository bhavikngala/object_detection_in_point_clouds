import torch
import torch.nn as nn
from networks.blocks import *
import numpy as np
import config as cnf
# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


class PointCloudDetector2(nn.Module):

	# res_block_layers = list if number of channels in the first conv layer of each res_block
	# up_sample_layers = list of tuple of number of channels input deconv and conv layers
	# up_sample_deconv = tuplr of (dilation, stride, padding, output_padding) for deconvolution in upsampling
	def __init__(self, res_block_layers, up_sample_layers, up_sample_deconv, mean=None, std=None):

		super(PointCloudDetector2, self).__init__()

		# keep padding in mind since at all convolutions are "SAME"
		# except for layers where downsampling happens

		self.conv1 = nn.Conv2d(in_channels=cnf.in_channels, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
		# self.bn2 = nn.BatchNorm2d(32)

		self.res_block1 = Bottleneck3FullPreActivation(in_channels = 32, out_channels=res_block_layers[0])
		self.res_block2 = Bottleneck6FullPreActivation(in_channels = 4 * res_block_layers[0], out_channels=res_block_layers[1])
		self.res_block3 = Bottleneck6FullPreActivation(in_channels = 4 * res_block_layers[1], out_channels=res_block_layers[2])
		self.res_block4 = Bottleneck3FullPreActivation(in_channels = 4 * res_block_layers[2], out_channels=res_block_layers[3])

		self.conv4 = nn.Conv2d(in_channels=4*res_block_layers[3], out_channels=196, kernel_size=1, bias=True)
		
		self.upsample1 = Upsample_2(in_channels = up_sample_layers[0], out_channels = 128, args = up_sample_deconv[0])
		
		self.upsample2 = Upsample_2(in_channels = up_sample_layers[1], out_channels = 96, args = up_sample_deconv[1])
		
		self.conv5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=True)
		
		self.conv6 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=True)
		
		self.conv7 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=True)
		
		self.conv8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=True)
		
		# sigmoid activation
		self.conv_cla = nn.Conv2d(in_channels=96, out_channels=1, kernel_size=3, padding=1, bias=True)

		# check activation function from SSD code
		self.conv_loc = nn.Conv2d(in_channels=96, out_channels=6, kernel_size=3, padding=1, bias=True)

		self.relu = nn.ReLU(inplace=True)
		self.cla_act = nn.Sigmoid()

		if mean is not None:
			self.unstandarize = UnStandarizeLayer(mean, std)
		else:
			self.unstandarize = None

		# initialization for each layer of network
		self.layerInit()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		# x = self.bn2(x)
		# x = self.relu(x)

		x = self.res_block1(x)
		
		res_2 = self.res_block2(x)
		
		res_3 = self.res_block3(res_2)
		
		x = self.res_block4(res_3)
		
		x = self.conv4(x)
		
		x = self.upsample1(x, res_3)
		
		x = self.upsample2(x, res_2)
		
		x = self.conv5(x)
		x = self.relu(x)

		x = self.conv6(x)
		x = self.relu(x)

		x = self.conv7(x)
		x = self.relu(x)

		x = self.conv8(x)
		x = self.relu(x)

		cla = self.conv_cla(x)
		cla = self.cla_act(cla)

		loc = self.conv_loc(x)

		if self.unstandarize is not None:
			loc = self.unstandarize(loc)

		return cla, loc

	def layerInit(self):
		self.conv1.weight.data.normal_(0.0, 0.01)
		self.bn1.weight.data.normal_(1.0, 0.02)
		self.bn1.bias.data.fill_(0)
		self.conv2.weight.data.normal_(0.0, 0.01)

		self.res_block1.apply(resnetInit)
		self.res_block2.apply(resnetInit)
		self.res_block3.apply(resnetInit)
		self.res_block4.apply(resnetInit)
		self.upsample1.apply(upsampleInit)
		self.upsample2.apply(upsampleInit)

		self.conv5.weight.data.normal_(0.0, 0.01)
		self.conv5.bias.data.fill_(0)
		self.conv6.weight.data.normal_(0.0, 0.01)
		self.conv6.bias.data.fill_(0)
		self.conv7.weight.data.normal_(0.0, 0.01)
		self.conv7.bias.data.fill_(0)
		self.conv8.weight.data.normal_(0.0, 0.01)
		self.conv8.bias.data.fill_(0)

		self.conv_cla.weight.data.normal_(0.0, 0.01)
		t = torch.tensor((1-0.01)/0.01, dtype=torch.float32)
		self.conv_cla.bias.data.fill_(t.log())

		self.conv_loc.weight.data.normal_(0.0, 0.01)
		self.conv_loc.bias.data.fill_(0)


def resnetInit(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
	elif isinstance(m, nn.BatchNorm2d):
		nn.init.constant_(m.weight, 1)
		nn.init.constant_(m.bias, 0)


def upsampleInit(m):
	if isinstance(m, nn.Conv2d):
		m.weight.data.normal_(0.0, 0.01)
		m.bias.data.fill_(0)		
	elif isinstance(m, nn.ConvTranspose2d):
		m.weight.data.normal_(0.0, 0.01)
		m.bias.data.fill_(0)