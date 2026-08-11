import torch
import torch.nn as nn


class InverseResnet50:
    def __init__(self):
        raise NotImplementedError



class LinearDecoder(nn.Module):
    """
    Inverse backbone for fixed image sizes: bx256x15x20 -> bx3x480x640 (b x c x h x w)
    """

    # RESTNET50: 3x480x640 -conv> -pooling> 64x120x160 -block1> 256x120x160 -block2> 512x60x80 -block3> 1024x30x40
    # -block4> 2048x15x20
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(256, 3, kernel_size=32, stride=32)
        self.sig_1 = nn.Sigmoid()

    def forward(self, x):
        # x = x.transpose(1, 2).unflatten(dim=2, sizes=(14, 14)
        x = self.up_conv_1(x)
        x = self.sig_1(x)
        return x


class LinearDecoderEnhanced(nn.Module):
    """
    Inverse backbone for fixed image sizes: bx256x15x20 -> bx3x480x640 (b x c x h x w)
    """

    # RESTNET50: 3x480x640 -conv> -pooling> 64x120x160 -block1> 256x120x160 -block2> 512x60x80 -block3> 1024x30x40
    # -block4> 2048x15x20
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1)
        self.relu_2 = nn.ReLU()

        self.up_conv_3 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1)
        self.relu_3 = nn.ReLU()

        self.up_conv_4 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1)
        self.relu_4 = nn.ReLU()

        self.up_conv_5 = nn.ConvTranspose2d(256, 3, kernel_size=32, stride=32)
        self.sig_5 = nn.Sigmoid()


    def forward(self, x):
        # x = x.transpose(1, 2).unflatten(dim=2, sizes=(14, 14))
        x = self.up_conv_1(x)
        x = self.relu_1(x)
        x = self.up_conv_2(x)
        x = self.relu_2(x)
        x = self.up_conv_3(x)
        x = self.relu_3(x)
        x = self.up_conv_4(x)
        x = self.relu_4(x)
        x = self.up_conv_5(x)
        x = self.sig_5(x)
        return x


class SimpleConvolutionalDecoder(nn.Module):
    """
    Inverse backbone for fixed image sizes: bx256x15x20 -> bx3x480x640 (b x c x h x w)
    """

    # RESTNET50: 3x480x640 -conv> -pooling> 64x120x160 -block1> 256x120x160 -block2> 512x60x80 -block3> 1024x30x40
    # -block4> 2048x15x20

    def __init__(self):
        super().__init__()
        self.conv_t_1 = nn.ConvTranspose2d(256, 2048, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.relu_1 = nn.ReLU()  # bx2048x15x20

        self.conv_t_2 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_2 = nn.ReLU()  # 1024x30x40

        self.conv_t_3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_3 = nn.ReLU()  # 512x60x80

        self.conv_t_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_4 = nn.ReLU()  # 256x120x160

        self.conv_t_5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_5 = nn.ReLU()  # 64x240x320

        self.conv_t_6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu_6 = nn.ReLU()  # 64x480x320

        self.conv_t_7 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.sig_7 = nn.Sigmoid()  # 3x128x128
        # self.tan_7 = nn.Tanh()  # 3x128x128

    def forward(self, x):
        x = self.relu_1(self.conv_t_1(x))
        x = self.relu_2(self.conv_t_2(x))
        x = self.relu_3(self.conv_t_3(x))
        x = self.relu_4(self.conv_t_4(x))
        x = self.relu_5(self.conv_t_5(x))
        x = self.relu_6(self.conv_t_6(x))
        x = self.sig_7(self.conv_t_7(x))
        return x


class BatchNormalizedConvolutionalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(256, 2048, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.bn_1 = nn.BatchNorm2d(2048)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_2 = nn.BatchNorm2d(1024)
        self.relu_2 = nn.ReLU()

        self.up_conv_3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_3 = nn.BatchNorm2d(512)
        self.relu_3 = nn.ReLU()

        self.up_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.relu_4 = nn.ReLU()

        self.up_conv_5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.relu_5 = nn.ReLU()

        self.up_conv_6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_6 = nn.BatchNorm2d(64)
        self.relu_6 = nn.ReLU()

        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.sig_final = nn.Sigmoid()

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.up_conv_1(x)))
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.relu_4(self.bn_4(self.up_conv_4(x)))
        x = self.relu_5(self.bn_5(self.up_conv_5(x)))
        x = self.relu_6(self.bn_6(self.up_conv_6(x)))
        x = self.sig_final(self.final_conv(x))
        return x


class EnhancedBatchNormalizedConvolutionalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(256, 2048, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.bn_1 = nn.BatchNorm2d(2048)
        self.relu_1 = nn.ReLU()

        # Gradual increase in channels
        # self.up_conv_inter_1 = nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=1, padding=1, output_padding=0)
        # self.bn_inter_1 = nn.BatchNorm2d(1024)
        # self.relu_inter_1 = nn.ReLU()

        # self.up_conv_inter_2 = nn.ConvTranspose2d(1024, 2048, kernel_size=1, stride=1, padding=0, output_padding=0)
        # self.bn_inter_2 = nn.BatchNorm2d(2048)
        # self.relu_inter_2 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(2048, 1792, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn_2 = nn.BatchNorm2d(1792)
        self.relu_2 = nn.ReLU()

        self.up_conv_2_2 = nn.ConvTranspose2d(1792, 1536, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.bn_2_2 = nn.BatchNorm2d(1536)
        self.relu_2_2 = nn.ReLU()

        self.up_conv_2_3 = nn.ConvTranspose2d(1536, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_2_3 = nn.BatchNorm2d(1024)
        self.relu_2_3 = nn.ReLU()

        self.up_conv_3 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_3 = nn.BatchNorm2d(512)
        self.relu_3 = nn.ReLU()

        self.up_conv_4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_4 = nn.BatchNorm2d(256)
        self.relu_4 = nn.ReLU()

        self.up_conv_5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_5 = nn.BatchNorm2d(128)
        self.relu_5 = nn.ReLU()

        self.up_conv_6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_6 = nn.BatchNorm2d(64)
        self.relu_6 = nn.ReLU()

        self.final_conv = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.sig_final = nn.Sigmoid()

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.up_conv_1(x)))
        # x = self.relu_inter_1(self.bn_inter_1(self.up_conv_inter_1(x)))
        # x = self.relu_inter_2(self.bn_inter_2(self.up_conv_inter_2(x)))
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_2_2(self.bn_2_2(self.up_conv_2_2(x)))
        x = self.relu_2_3(self.bn_2_3(self.up_conv_2_3(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.relu_4(self.bn_4(self.up_conv_4(x)))
        x = self.relu_5(self.bn_5(self.up_conv_5(x)))
        x = self.relu_6(self.bn_6(self.up_conv_6(x)))
        x = self.sig_final(self.final_conv(x))
        return x


if __name__ == '__main__':
    tensor = torch.rand(size=(10, 256, 15, 20))
    inv_backbone = LinearDecoderEnhanced()
    inv_backbone(tensor)
