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
