import torch
import torch.nn as nn
import torch.nn.functional as F


def bottleneck(in_channels, out_channels, stride):
	return nn.Sequential(
		nn.BatchNorm2d(in_channels[0]),
		nn.ReLU(),
		nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
		nn.BatchNorm2d(in_channels[1]),
		nn.ReLU(),
		nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False),
		nn.BatchNorm2d(in_channels[2]),
		nn.ReLU(),
		nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, bias=False)
	)


class Bottleneck3FullPreActivation(nn.Module):
	expansion = 4

	# input dim : c x 800 x 700
	# output dim: c x 400 x 350
	def __init__(self, in_channels, out_channels):
		super(Bottleneck3FullPreActivation, self).__init__()

		# using pre-normalization and pre-activation
		in_channels_ = [in_channels, out_channels, out_channels]
		out_channels_ = [out_channels, out_channels, self.expansion*out_channels]

		self.conv1_skip = nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=2, bias=False)
		self.bottleneck1 = bottleneck(in_channels_, out_channels_, stride=2)
		

	def forward(self, x):
		res = self.conv1_skip(x)
		return res + self.bottleneck1(x)


class Bottleneck6FullPreActivation(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck6FullPreActivation, self).__init__()

		in_channels1_ = [in_channels, out_channels, out_channels]
		in_channels2_ = [self.expansion*out_channels, out_channels, out_channels]
		out_channels_ = [out_channels, out_channels, self.expansion*out_channels]

		self.conv1_skip = nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=2, bias=False)
		self.bottleneck1 = bottleneck(in_channels1_, out_channels_, stride=2)
		self.bottleneck2 = bottleneck(in_channels2_, out_channels_, stride=1)


	def forward(self, x):
		res = self.conv1_skip(x)
		res = res + self.bottleneck1(x) 
		return res + self.bottleneck2(res)


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