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
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=2, padding=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)
		
		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.bn1(x)
		x = self.relu(x)

		res = self.conv1_skip(x)

		x = self.conv1(x)
		x = self.bn2(x)
		x = self.conv2(self.relu(x))

		out = x+res

		return out


class Bottleneck_3_1(nn.Module):
	expansion = 4

	# input dim : c x 800 x 700
	# output dim: c x 400 x 350
	def __init__(self, in_channels, out_channels):
		super(Bottleneck_3_1, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)
		self.bn1_skip = nn.BatchNorm2d(out_channels*self.expansion)

		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):

		res = self.conv1_skip(x)
		res = self.bn1_skip(res)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(self.relu(x))
		x = self.bn2(x)

		out = x+res
		out = self.relu(out)

		return out


class Bottleneck_6_0(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck_6_0, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn3 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

		self.bn4 = nn.BatchNorm2d(out_channels)
		self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

		# self.bn5 = nn.BatchNorm2d(out_channels*self.expansion)
		self.bn5 = nn.BatchNorm2d(out_channels)
		self.conv5 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)
		# self.bn1_skip = nn.BatchNorm2d(out_channels*self.expansion)

		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.bn1(x)
		x = self.relu(x)
		# res = self.bn1_skip(res)
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

		out = x+res

		return out


class Bottleneck_6_1_0(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck_6_1_0, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

		self.bn3 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)
		
		self.bn4 = nn.BatchNorm2d(out_channels*self.expansion)
		self.conv4 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=1, bias=False)

		self.bn5 = nn.BatchNorm2d(out_channels*self.expansion)
		self.conv5 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=1, bias=False)

		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.bn1(x)
		x = self.relu(x)
		# res = self.bn1_skip(res)
		res = self.conv1_skip(x)

		x = self.conv1(x)
		x = self.bn2(x)
		x = self.conv2(self.relu(x))
		x = self.bn3(x)
		x = self.conv3(self.relu(x))
		x = x + res
		
		res = self.bn4(x)
		res = self.relu(res)
		x = self.conv4(res)
		x = self.bn5(x)
		x = self.conv5(self.relu(x))

		out = x+res

		return out


class Bottleneck_6_1_1(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels):
		super(Bottleneck_6_1_1, self).__init__()

		# using pre-normalization and pre-activation
		# TODO: switch stride=2 between conv1 and conv2 and check results
		# self.bn1 = nn.BatchNorm2d(in_channels)
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

		self.conv1_skip = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=2, bias=False)
		self.bn1_skip = nn.BatchNorm2d(out_channels*self.expansion)

		self.conv4 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=1, bias=False)
		self.bn4 = nn.BatchNorm2d(out_channels*self.expansion)

		self.conv5 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=1, bias=False)
		self.bn5 = nn.BatchNorm2d(out_channels*self.expansion)

		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		res = self.conv1_skip(x)
		res = self.bn1_skip(res)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(self.relu(x))
		x = self.bn3(x)
		
		x = x + res
		x = self.relu(x)
		
		res = x

		x = self.conv4(x)
		x = self.bn4(x)
		x = self.relu(x)
		
		x = self.conv5(x)
		x = self.bn5(x)

		out = x+res
		out = self.relu(out)

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
		out = d+res

		return out


class UnStandarizeLayer(nn.Module):
	'''
	Changes the view of the input tensor
	Then un standarizes the input tensor by given mean and std
	'''
	def __init__(self, mean, std):
		super(UnStandarizeLayer, self).__init__()

		self.register_buffer('mean', mean)
		self.register_buffer('std', std)

	def forward(self, X):
		m, c, h, w = X.size()
		X = X.permute(0, 2, 3, 1).contiguous().view(m, w*h, c)
		X = X*self.std + self.mean
		return X