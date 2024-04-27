import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMorphoTM(nn.Module):

    def __init__(self, input_dim, init_dim):
        super(DeepMorphoTM, self).__init__()

        self.pool = nn.MaxPool2d(2)
        ## encode ##
        self.conv1_1 = nn.Conv2d(input_dim, init_dim, 3, padding='same')       # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1_2 = nn.Conv2d(init_dim, init_dim, 3, padding='same')
        # 256 - pool1 - 128
        self.conv2_1 = nn.Conv2d(init_dim, init_dim*2, 3, padding='same')
        self.conv2_2 = nn.Conv2d(init_dim*2, init_dim*2, 3, padding='same')
        # 128 - pool2 - 64
        self.conv3_1 = nn.Conv2d(init_dim*2, init_dim*4, 3, padding='same')
        self.conv3_2 = nn.Conv2d(init_dim*4, init_dim*4, 3, padding='same')
        # 64 - pool3 - 32
        self.conv4_1 = nn.Conv2d(init_dim*4, init_dim*8, 3, padding='same')
        self.conv4_2 = nn.Conv2d(init_dim*8, init_dim*8, 3, padding='same')
        # 32 - pool4 - 16
        self.conv5_1 = nn.Conv2d(init_dim*8, init_dim*16, 3, padding='same')
        self.conv5_2 = nn.Conv2d(init_dim*16, init_dim*16, 3, padding='same')
        # 16 - pool5 - 8
        self.conv6_1 = nn.Conv2d(init_dim*16, init_dim*32, 3) # 6
        self.conv6_2 = nn.Conv2d(init_dim*32, init_dim*32, 3) # 4
        self.conv6_3 = nn.Conv2d(init_dim*32, init_dim*32, 3) # 2
        self.conv6_4 = nn.Conv2d(init_dim*32, init_dim*32, 2) # 1
        ## add scalar inputs ##
        ## decode ##
        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        self.transconv6_1 = nn.ConvTranspose2d(init_dim*32 + 1, init_dim*32, 2, 2) # 4 times; 16
        self.transconv6_2 = nn.ConvTranspose2d(init_dim*32, init_dim*32, 2, 2)
        self.transconv6_3 = nn.ConvTranspose2d(init_dim*32, init_dim*32, 2, 2)
        self.transconv6_4 = nn.ConvTranspose2d(init_dim*32, init_dim*32, 2, 2)
        #
        self.dc_conv5_1 = nn.Conv2d(init_dim*48, init_dim*32, 3, padding='same')
        self.dc_conv5_2 = nn.Conv2d(init_dim*32, init_dim*32, 3, padding='same')
        self.transconv_p4 = nn.ConvTranspose2d(init_dim*32, init_dim*32, 2, 2) # 32
        #
        self.dc_conv4_1 = nn.Conv2d(init_dim*40, init_dim*16, 3, padding='same')
        self.dc_conv4_2 = nn.Conv2d(init_dim*16, init_dim*16, 3, padding='same')
        self.transconv_p3 = nn.ConvTranspose2d(init_dim*16, init_dim*16, 2, 2) # 64
        #
        self.dc_conv3_1 = nn.Conv2d(init_dim*20, init_dim*8, 3, padding='same')
        self.dc_conv3_2 = nn.Conv2d(init_dim*8, init_dim*8, 3, padding='same')
        self.transconv_p2 = nn.ConvTranspose2d(init_dim*8, init_dim*8, 2, 2) # 128
        #
        self.dc_conv2_1 = nn.Conv2d(init_dim*10, init_dim*4, 3, padding='same')
        self.dc_conv2_2 = nn.Conv2d(init_dim*4, init_dim*4, 3, padding='same')
        self.transconv_p1 = nn.ConvTranspose2d(init_dim*4, init_dim*4, 2, 2) # 256
        #
        self.dc_conv1_1 = nn.Conv2d(init_dim*5, init_dim*2, 3, padding='same')
        self.dc_conv1_2 = nn.Conv2d(init_dim*2, init_dim*2, 3, padding='same')
        self.dc_conv1_3 = nn.Conv2d(init_dim*2, init_dim, 3, padding='same')
        ## out ##
        self.out1 = nn.Conv2d(init_dim, 2, 3, padding='same')


    def forward(self, x, moduli):
        x1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        x2 = F.relu(self.conv2_2(F.relu(self.conv2_1(self.pool(x1)))))
        x3 = F.relu(self.conv3_2(F.relu(self.conv3_1(self.pool(x2)))))
        x4 = F.relu(self.conv4_2(F.relu(self.conv4_1(self.pool(x3)))))
        x5 = F.relu(self.conv5_2(F.relu(self.conv5_1(self.pool(x4)))))
        x = F.relu(self.conv6_4(F.relu(self.conv6_3(F.relu(self.conv6_2(F.relu(self.conv6_1(self.pool(x5)))))))))
        x = torch.cat((x, moduli), 1)
        x = F.relu(self.transconv6_4(F.relu(self.transconv6_3(F.relu(self.transconv6_2(F.relu(self.transconv6_1(x))))))))
        x = torch.cat((x5, x), 1)
        x = self.transconv_p4(F.relu(self.dc_conv5_2(F.relu(self.dc_conv5_1(x)))))
        x = torch.cat((x4, x), 1)
        x = self.transconv_p3(F.relu(self.dc_conv4_2(F.relu(self.dc_conv4_1(x)))))
        x = torch.cat((x3, x), 1)
        x = self.transconv_p2(F.relu(self.dc_conv3_2(F.relu(self.dc_conv3_1(x)))))
        x = torch.cat((x2, x), 1)
        x = self.transconv_p1(F.relu(self.dc_conv2_2(F.relu(self.dc_conv2_1(x)))))
        x = torch.cat((x1, x), 1)
        x = F.relu(self.dc_conv1_3((F.relu(self.dc_conv1_2(F.relu(self.dc_conv1_1(x)))))))

        return self.out1(x)

