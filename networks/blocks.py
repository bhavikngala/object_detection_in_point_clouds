import torch
import torch.nn as nn
import torch.nn.functional as F

# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class Bottleneck_3_0(nn.Module):
	expansion = 4

	# input dim : c x 800 x 700
	# output dim: c x 400 x 350
	def __init__(self, in_channels, out_channels):
		super(Bottleneck_3_0, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=2, padding=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		res = self.conv1_skip(x)

		x = self.conv1(x)
		x = self.bn2(x)
		x = self.conv2(self.relu(x))

		return x + res

class Bottleneck_4_0(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck_4_0, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn3 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		res = self.conv1_skip(x)

		x = self.conv1(x)
		x = self.bn2(x)
		x = self.conv2(self.relu(x))
		x = self.bn3(x)
		x = self.conv3(self.relu(x))

		return x + res

class Bottleneck_6_0(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck_6_0, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn3 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

		self.bn4 = nn.BatchNorm2d(out_channels)
		self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

		self.bn5 = nn.BatchNorm2d(out_channels)
		self.conv5 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		res = self.conv1_skip(x)

		x = self.conv1(x)
		x = self.bn2(x)
		x = self.conv2(self.relu(x))
		x = self.bn3(x)
		x = self.conv3(self.relu(x))
		x = self.bn4(x)
		x = self.conv4(self.relu(x))
		x = self.bn5(x)
		x = self.conv5(self.relu(x))

		return x + res

class Upsample(nn.Module):
	'''
	Upsamples the input sample by a factor of 2.
	First the input is upsample by bilinear interpolation.
	Then, convolution is applied to the interpolated image.
	Reference: https://distill.pub/2016/deconv-checkerboard/
	'''
	def __init__(self, in_channels, out_channels):
		super(Upsample, self).__init__()
		self.conv_upsample = nn.Conv2d(in_channels[0], out_channels, kernel_size=3, padding=1, bias=False)
		self.conv1 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, bias=False)

	def forward(self, featureMapToUpsample, originalFeatureMap):
		u = F.interpolate(featureMapToUpsample, scale_factor=2, mode='bilinear')
		u = self.conv_upsample(u)

		x = self.conv1(originalFeatureMap)

		return x + u

# for new variants of bottleneck change names here
Bottleneck_3 = Bottleneck_3_0
Bottleneck_4 = Bottleneck_4_0
Bottleneck_6 = Bottleneck_6_0
Upsample = Upsample