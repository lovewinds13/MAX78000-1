import torch
import torch.nn as nn

import ai8x

class AI85Net_GARBAGE(nn.Module):
    """
    5-Layer CNN that uses max parameters in AI84
    """
    def __init__(self, num_classes=6, num_channels=3, dimensions= (3, 224, 224),
                 planes=32, pool=2, fc_inputs=256, bias=False, **kwargs):
        super().__init__()

        # Limits
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        # Keep track of image dimensions so one constructor works for all image sizes
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, stride=1, padding=1, bias=True, **kwargs)
        self.conv2 = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=2, padding=1, bias=True, **kwargs)
        self.conv3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=2, padding=1, bias=True, **kwargs)
        self.conv4 = ai8x.FusedMaxPoolConv2d(64, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=True, **kwargs)
        # self.conv4 = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, pool_size=2, pool_stride=2,
        #                                            stride=1, padding=1, bias=bias, **kwargs)

        self.pooling=ai8x.MaxPool2d(8)
        
        self.avg = nn.AdaptiveAvgPool2d(1)#自适应平均池化
        
        self.fc1 = ai8x.FusedLinearReLU(128, 128, bias=True, **kwargs)
        self.fc = ai8x.Linear(128, num_classes, bias=True, wide=True, **kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.size())
        x = self.pooling(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc(x)

        return x

    
def ai85net_garbage(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_GARBAGE(**kwargs)

models = [
    {
        'name': 'ai85net_garbage',
        'min_input': 1,
        'dim': 1,
    },

]
