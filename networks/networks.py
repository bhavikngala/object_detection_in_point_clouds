import torch
import torch.nn as nn
from networks.blocks import *
import numpy as np
import math
import config as cnf
# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


class PointCloudDetector2(nn.Module):

	# res_block_layers = list if number of channels in the first conv layer of each res_block
	# up_sample_layers = list of tuple of number of channels input deconv and conv layers
	# up_sample_deconv = tuplr of (dilation, stride, padding, output_padding) for deconvolution in upsampling
	def __init__(self, res_block_layers, up_sample_layers, up_sample_deconv):

		super(PointCloudDetector2, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=cnf.in_channels, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(32)

		self.inplanes = 32
		self.res_block1 = self._make_layer(Bottleneck, 24, 1, stride=2)
		self.res_block2 = self._make_layer(Bottleneck, 48, 2, stride=2)
		self.res_block3 = self._make_layer(Bottleneck, 64, 2, stride=2)
		self.res_block4 = self._make_layer(Bottleneck, 96, 1, stride=2)

		self.conv4 = nn.Conv2d(in_channels=self.inplanes, out_channels=196, kernel_size=1, bias=True)
		
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

		# initialization for each layer of network
		self.layerInit()

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn1(x)

		x = self.conv2(x)
		x = self.relu(x)
		x = self.bn2(x)

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

		return cla, loc


	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)


	def layerInit(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

		prior = 0.01
		self.conv_cla.bias.data.fill_(-math.log((1.0-prior)/prior))