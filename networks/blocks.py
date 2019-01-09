import torch
import torch.nn as nn
import torch.nn.functional as F

# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class Bottleneck3FullPreActivation(nn.Module):
	expansion = 4

	# input dim : c x 800 x 700
	# output dim: c x 400 x 350
	def __init__(self, in_channels, out_channels):
		super(Bottleneck3FullPreActivation, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn1_skip = nn.BatchNorm2d(in_channels)
		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)

		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):

		# res = self.bn1_skip(x)
		# res = self.relu(res)
		res = self.conv1_skip(res)

		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv1(x)

		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv2(x)

		out = x+res

		return out


class Bottleneck6FullPreActivation(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck6FullPreActivation, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		# self.bn1 = nn.BatchNorm2d(in_channels)
		self.bn1_skip = nn.BatchNorm2d(in_channels)
		self.conv1_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
		
		self.bn3 = nn.BatchNorm2d(in_channels)
		self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

		self.bn4 = nn.BatchNorm2d(out_channels)
		self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn5 = nn.BatchNorm2d(out_channels)
		self.conv5 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)


		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):
		# res = self.bn1_skip(x)
		# res = self.relu(res)
		res = self.conv1_skip(res)

		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv1(x)

		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv2(x)

		res = x + res

		x = self.bn3(res)
		x = self.relu(x)
		x = self.conv3(x)
		
		x = self.bn4(x)
		x = self.relu(x)
		x = self.conv4(x)
		
		x = self.bn5(x)
		x = self.relu(x)
		x = self.conv5(x)

		res = x+res

		return res


class Upsample_2(nn.Module):
	'''
	Upsampling block as described in the PIXOR paper.
	First apply deconvolution to input, apply convolution 
	to res connection, and perform element wise addition.

	Requires: in_channels
			: out_channels
			: output_size
			: args - (dilation, stride, padding, output_padding) for deconv layer
	'''
	def __init__(self, in_channels, out_channels, args):
		super(Upsample_2, self).__init__()

		dilation, stride, padding, output_padding = args
		self.deconv1 = nn.ConvTranspose2d(
			in_channels[0],
			out_channels,
			kernel_size=3,
			stride=stride,
			padding=padding,
			output_padding=output_padding,
			groups=1,
			bias=True,
			dilation=dilation
		)
		# self.bn_deconv1 = nn.BatchNorm2d(out_channels)

		self.conv1 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, bias=True)
		# self.bn1 = nn.BatchNorm2d(out_channels)

		# self.relu = nn.ReLU(inplace=True)

	def forward(self, featureMapToUpsample, originalFeatureMap):
		d = self.deconv1(featureMapToUpsample)
		# d = self.bn_deconv1(d)

		res = self.conv1(originalFeatureMap)
		# res = self.bn1(res)

		# out = self.relu(d+res)
		out = d+res

		return out