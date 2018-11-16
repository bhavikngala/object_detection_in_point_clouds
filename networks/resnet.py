import torch.nn as nn
from networks.blocks import UnStandarizeLayer

'''
This is official resnet implementation from pytorch developers.
All credit for this code goes to the authors.
Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
I have modifications in parameters to suite my network.
'''

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, layers=[2, 2, 2, 2], mean=None, std=None):
        super(ResNet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(36, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3])

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(96)
        self.conv8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(96)
        self.conv9 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(96)
        self.conv10 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(96)

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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

        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)

        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)

        cla = self.conv_cla(x)
        cla = self.cla_act(cla)

        loc = self.conv_loc(x)

        if self.unstandarize is not None:
            loc = self.unstandarize(loc)

        return cla, loc