import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class my_resnet50(nn.Module):
    def __init__(self):
        super(my_resnet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()

        resnet50_fc = list(resnet50.children())[:-1]
        self.resnet50_fc = nn.Sequential(*resnet50_fc)
        self.resnet50_fc.eval()

        resnet50_s = list(resnet50.children())[-1:]
        self.resnet50_s = nn.Sequential(*resnet50_s)
        self.resnet50_s.eval()
    
    def forward(self, x):
        with torch.no_grad():
            resnet_fc = self.resnet50_fc(x).squeeze()
            resnet_s = self.resnet50_s(resnet_fc).squeeze()
        return dict(fc=resnet_fc,s=resnet_s)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            #nn.Flatten()
            )

    def forward(self, input_):
        return self.net(input_)

class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 4),
            nn.ReLU(inplace=True),
            )
    def forward(self, input_):
        return self.net(input_)
    
class Decoder2(nn.Module):
    def __init__(
        self,
        input_channels = 128
    ):
        super(Decoder2, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, padding=1),
            )
    def forward(self, input_):
        return self.net(input_)
