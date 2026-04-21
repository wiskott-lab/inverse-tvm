import timm
import torch
import torch.nn as nn

class InverseInputProjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InverseResnetBlock(nn.Module):

    def __init__(self, in_channels,  num_blocks=4, out_channels=None, upsample=True, last_output=False):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels // 2
        self.layers = nn.Sequential()
        for i in range(num_blocks - 1):
            self.layers.append(nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU()))

        if last_output:
            self.last_dumb_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=4, padding=2, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid())
        else:
            if upsample:
                stride, output_padding = 2, 1
            else:
                stride, output_padding = 1, 0
            self.last_dumb_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                   padding=1, output_padding=output_padding),
                nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        out = self.last_dumb_layer(x)
        return out




# outwith torch.no_grad():
#     resnet = timm.create_model(model_name='resnet50', pretrained=True)
#     a = (torch.rand(size=(10, 3, 480, 640)) - 0.5) * 2.2
#     int_reps = get_int_reps_from_resnet(resnet, a)
#
#     inverse_blocks = [InverseResnetBlock(in_channels=2048), InverseResnetBlock(in_channels=1024), InverseResnetBlock(in_channels=512), InverseResnetBlock(in_channels=256, out_channels=64, upsample=False), InverseResnetBlock(in_channels=64, out_channels=3, last_output=True)]
#     y = int_reps[-1]
#     for block in inverse_blocks:
#         y = block(y)
#
#
#
#     pass
#
