import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
                 nn.BatchNorm2d(64),
                 nn.ReLU(True)]

        # downsampling
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True)]

        # resnet blocks
        for i in range(n_blocks):
            model.append(ResnetBlock(256))

        # upsampling
        model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, bias=False),
                  # nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                  nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, bias=False),
                  # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, 3, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


def UnetEncodeBlock(in_channels, out_channels, last=False):
    if not last:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True)
        )


def UnetDecodeBlock(in_channels, out_channels, last=False):
    if not last:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )


class UnetGenerator(nn.Module):
    def __init__(self, n_steps):
        super(UnetGenerator, self).__init__()

        self.encode = nn.ModuleList([
            UnetEncodeBlock(3, 64),
            UnetEncodeBlock(64, 128),
            UnetEncodeBlock(128, 256),
            UnetEncodeBlock(256, 512),
        ])

        for _ in range(n_steps - 5):
            self.encode.append(UnetEncodeBlock(512, 512))
        self.encode.append(UnetEncodeBlock(512, 512, True))

        self.decode = nn.ModuleList([
            UnetDecodeBlock(512 // 2, 512),
        ])
        for _ in range(n_steps - 5):
            self.decode.append(UnetDecodeBlock(512, 512))

        self.decode.extend([
            UnetDecodeBlock(512, 256),
            UnetDecodeBlock(256, 128),
            UnetDecodeBlock(128, 64),
            UnetDecodeBlock(64, 3, True)
        ])

    def forward(self, x):
        encode_x = []
        for layer in self.encode[:-1]:
            x = layer(x)
            encode_x.append(x)
        x = self.encode[-1](x)
        
        x = self.decode[0](x)
        for layer, prev_x in zip(self.decode[1:], encode_x[::-1]):
            x = layer(torch.cat((prev_x, x), 1))
        return x
