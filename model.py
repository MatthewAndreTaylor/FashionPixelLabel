import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    scale_factor = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, dilation=1
    ):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.scale_factor, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, groups=1, width_per_group=64, pretrained=True):
        super(ResNet, self).__init__()
        planes = [int(width_per_group * groups * 2**i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(
            3, planes[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, planes[0], layers[0], groups=groups)
        self.layer2 = self.make_layer(
            block, planes[1], layers[1], stride=2, groups=groups
        )
        self.layer3 = self.make_layer(
            block, planes[2], layers[2], stride=2, groups=groups
        )
        self.layer4 = self.make_layer(
            block, planes[3], layers[3], stride=1, groups=groups, dilations=[2, 4, 8]
        )
        self._init_weight()
        if pretrained:
            self._load_from_pretrained()

    def make_layer(self, block, planes, blocks, stride=1, groups=1, dilations=None):
        if dilations is None:
            dilations = [1] * blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.scale_factor:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.scale_factor,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.scale_factor),
            )
        layers = [
            block(
                self.inplanes, planes, stride, downsample, groups, dilation=dilations[0]
            )
        ]
        self.inplanes = planes * block.scale_factor
        layers.extend(
            [
                block(self.inplanes, planes, groups=groups, dilation=rate)
                for rate in dilations[1:]
            ]
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_from_pretrained(self):
        pretrain_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        )
        model_dict = {k: v for k, v in pretrain_dict.items() if k in self.state_dict()}
        self.load_state_dict(model_dict)


class ASPPLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation):
        super(ASPPLayer, self).__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel, padding=padding, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = ASPPLayer(in_channels, out_channels, 1, 0, 1)
        self.conv2 = ASPPLayer(in_channels, out_channels, 3, 6, 6)
        self.conv3 = ASPPLayer(in_channels, out_channels, 3, 12, 12)
        self.conv4 = ASPPLayer(in_channels, out_channels, 3, 18, 18)
        self.feature_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv = ASPPLayer(out_channels * 5, out_channels, 1, 0, 1)

    def forward(self, x):
        original_size = x.size()[2:]
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        features = F.interpolate(
            self.feature_pooling(x), size=original_size, mode="bilinear"
        )
        x = torch.cat([out1, out2, out3, out4, features], dim=1)
        x = self.conv(x)
        return x


class DeepLabv3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
        )
        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.resnet(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=original_size, mode="bilinear")
        return x


class DeepLabv3_plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabv3_plus, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
        )
        self.aspp = ASPP(2048, 256)
        self.one_by_one_conv = nn.Conv2d(2048, 256, 1)
        self.low_level_upsample = nn.Upsample(mode="bilinear", size=(600, 400))
        self.up_encoder = nn.Upsample(mode="bilinear", size=(600, 400))
        self.decoder_conv = nn.Conv2d(512, num_classes, 3)
        self.up_final = nn.Upsample(mode="bilinear", size=(600, 400))

    def forward(self, x):
        x = self.resnet(x)
        low_level_features = self.one_by_one_conv(x)
        low_level_features_up = self.low_level_upsample(low_level_features)
        encoder_out = self.aspp(x)
        encoder_out_up = self.up_encoder(encoder_out)
        combined = torch.cat((low_level_features_up, encoder_out_up), dim=1)
        out = self.decoder_conv(combined)
        up = self.up_final(out)
        return up


from crfseg import CRF


class CRFExtensionModule(nn.Module):
    def __init__(self, net, num_classes=2):
        super(CRFExtensionModule, self).__init__()
        self.model = net(num_classes)
        self.crf = CRF(num_classes, n_iter=5, smoothness_weight=1, smoothness_theta=1)

    def forward(self, x):
        x = self.model(x)
        x = self.crf(x)
        return x
