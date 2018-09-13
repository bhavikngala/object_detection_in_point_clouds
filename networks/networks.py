import torch
import torch.nn as nn
from blocks import Bottleneck_4, Bottleneck_6, Upsample

# resnet reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class PointCloudDetector(nn.Module):

	def __init__(self, res_block_layers, up_sample_layers):
		# res_block_layers = list if number of channels in the first conv layer of each res_block
		# up_sample_layers = list of tuple of number of channels input deconv and conv layers

		# keep padding in mind since at all convolutions are "SAME"
		# except for layers where downsampling happens
		self.conv1 = nn.Conv2d(in_channels=36, out_channels=32, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)

		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False)

		self.res_block1 = Bottleneck_4(in_channels = 32, out_channels=res_block_layers[0])
		self.res_block2 = Bottleneck_6(in_channels = 4 * res_block_layers[0], out_channels=res_block_layers[1])
		self.res_block3 = Bottleneck_6(in_channels = 4 * res_block_layers[0], out_channels=res_block_layers[2])
		self.res_block4 = Bottleneck_4(in_channels = 4 * res_block_layers[0], out_channels=res_block_layers[3])

		self.bn3 = nn.BatchNorm2d(res_block_layers[3])
		
		self.conv4 = nn.Conv2d(in_channels=res_block_layers[3], out_channels=196, kernel_size=1, bias=False)

		self.upsample1 = Upsample(in_channels = up_sample_layers[0], out_channels = 128)
		self.upsample2 = Upsample(in_channels = up_sample_layers[1], out_channels = 96)

		self.bn4 = nn.BatchNorm2d(96),
		
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

		x = self.res_block1(x)
		res_2 = self.res_block2(x)
		res_3 = self.res_block3(res_2)
		x = self.res_block4(res_3)

		x = self.bn3(x)
		x = self.relu(x)

		x = self.conv4(x)

		x = self.upsample1(x, res_3)
		x = self.upsample2(x, res_2)

		x = self.bn4(x)
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
		loc = self.loc_act(loc)

		# may want to reshape the tensors before returning
		return cla, loc