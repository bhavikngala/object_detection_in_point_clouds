import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.relu(out)
		out = self.bn1(out)

		out = self.conv2(out)
		out = self.relu(out)
		out = self.bn2(out)

		out = self.conv3(out)

		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity

		out = self.relu(out)
		out = self.bn3(out)

		return out


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
		return res.add_(d)