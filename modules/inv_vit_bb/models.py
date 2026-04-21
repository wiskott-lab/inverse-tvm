import torch.nn as nn
import torch


class InverseViTBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(768, 3, kernel_size=16, stride=16)
        self.sig_1 = nn.Sigmoid()


    def forward(self, x):
        x = x[:, 1:, :]  # cut class token
        x = x.transpose(1, 2).unflatten(dim=2, sizes=(14, 14))
        x = self.up_conv_1(x)
        x = self.sig_1(x)
        return x


class EnhancedVitBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()

        self.up_conv_3= nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_3 = nn.BatchNorm2d(16)
        self.relu_3 = nn.ReLU()

        self.up_conv_4 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.sig_4 = nn.Sigmoid()


    def forward(self, x):
        x = x[:, 1:, :]  # cut class token
        x = x.transpose(1, 2).unflatten(dim=2, sizes=(14, 14))
        x = self.relu_1(self.bn_1(self.up_conv_1(x)))
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.sig_4(self.up_conv_4(x))
        return x


class EnhancedInverseVitBackbonePos(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_2 = nn.BatchNorm2d(64)
        self.relu_2 = nn.ReLU()

        self.up_conv_3= nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_3 = nn.BatchNorm2d(16)
        self.relu_3 = nn.ReLU()

        self.up_conv_4 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.sig_4 = nn.Sigmoid()

        self.pos_emb = nn.Parameter(torch.empty(size=(1, 197, 768)))
        self.pos_emb.requires_grad = False


    def set_pos_emb(self, vit):
        self.pos_emb = vit.pos_embed
        self.pos_emb.requires_grad = False

    def forward(self, x):
        x = x - self.pos_emb
        x = x[:, 1:, :]  # cut class token
        x = x.transpose(1, 2).unflatten(dim=2, sizes=(14, 14))
        x = self.relu_1(self.bn_1(self.up_conv_1(x)))
        x = self.relu_2(self.bn_2(self.up_conv_2(x)))
        x = self.relu_3(self.bn_3(self.up_conv_3(x)))
        x = self.sig_4(self.up_conv_4(x))
        return x




class InverseViTBackboneEmbedder(nn.Module):

    def __init__(self, vit):
        super().__init__()
        conv_weight = vit.patch_embed.proj.weight.view(768, -1)
        self.conv_bias = vit.patch_embed.proj.bias
        self.inv_conv_weight =  torch.linalg.inv(conv_weight)
        self.pos_emb = vit.pos_embed

    def forward(self, x, *args, **kwargs):
        x = x - self.pos_emb
        x = x[:, 1:, :] # cut class token
        x = x - self.conv_bias
        x = torch.matmul(self.inv_conv_weight, x.transpose(1, 2))
        x = torch.nn.functional.fold(x, output_size=(224, 224), kernel_size=(16, 16), stride=16, padding=0)
        return x


if __name__ == '__main__':
    tensor = torch.rand(size=(10, 197, 768))
    bb = EnhancedVitBackbone()
    bb(tensor)