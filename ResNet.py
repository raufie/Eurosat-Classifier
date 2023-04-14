from torch import nn
import torch
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Sequential()
#         with the skip connection, the gradients flowing back will also contribute, in a case where conv layer messes up, that will at least
# be useful, so it acts as a regularization measure
        if stride!= 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        residual = self.skip_connection(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x+=residual
        
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 , 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(
        ResNetBlock(64, 64),
        ResNetBlock(64,64)
        )
        self.layer2 = nn.Sequential(
        ResNetBlock(64, 128, stride=2),
        ResNetBlock(128,128)
        )
        self.layer3 = nn.Sequential(
        ResNetBlock(128, 256, stride=2),
        ResNetBlock(256,256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
#         x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)       
        x = self.fc(x)
        return x
        