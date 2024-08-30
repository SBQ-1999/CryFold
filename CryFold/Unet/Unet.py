import torch
from torch import nn
import random
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=True,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)
class ConvBuildingBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm3d(out_channels,affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
    def forward(self,x:torch.Tensor):
        return self.activate_function(self.conv1(x)+self.shortcut_conv(x))

class ShortConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = nn.ReLU()
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,out_channels*self.scale,1,1,0,bias=False),nn.InstanceNorm3d(out_channels*self.scale,affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale,affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.scale,out_channels,1,1,0,bias=False),nn.InstanceNorm3d(out_channels,affine=True))
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )
    def forward(self, x):
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale,dim=1)

        y_list = []
        for ii,xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii-1](xi))
            else:
                y_list.append(self.conv_list[ii-1](xi+y_list[-1]))
        y = self.conv2(self.activate_class(self.norm1(torch.cat(y_list,dim=1))))
        y = self.activate_class(y+self.shortcut_conv(x))
        return y

class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet,self).__init__()
        self.shortconv0 = ShortConv(1,256)
        self.downsample1 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample2 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample3 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample4 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.main0 = self.main_layer(256, 2, 4)
        self.conv43 = ConvBuildingBlock(256+256,256)
        self.main1 = self.main_layer(256, 3, 4)
        self.conv32 = ConvBuildingBlock(256+256, 128)
        self.main2 = self.main_layer(128, 4, 4)
        self.conv21 = ConvBuildingBlock(256+128, 64)
        self.main3 = self.main_layer(64, 4, 4)
        self.conv10 = ConvBuildingBlock(256+64, 64)
        self.main4 = self.main_layer(64, 4, 4)
        self.conv14 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv3d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv3d(64, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv13 = nn.Conv3d(in_channels=32 * 3, out_channels=1, padding=1, kernel_size=3)
    def main_layer(self,input_channels,expansion,num_layers):
        layer=[]
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels,input_channels,scale=expansion))
        return nn.Sequential(*layer)
    #上采样
    #multi_scale_conv
    def forward(self,V):
        ds_0 = self.shortconv0(V)
        ds_1 = self.downsample1(ds_0)
        ds_2 = self.downsample2(ds_1)
        ds_3 = self.downsample3(ds_2)
        ds_4 = self.downsample4(ds_3)
        c4 = self.main0(ds_4)
        def upsample_add(f, g):
            g_shape = g.shape
            D, H, W = g_shape[2:]
            upsampled = nn.functional.interpolate(input=f, size=(D, H, W), mode='trilinear', align_corners=True)
            return torch.cat([upsampled,g],dim=1)
        c3 = self.main1(self.conv43(upsample_add(c4, ds_3)))
        c2 = self.main2(self.conv32(upsample_add(c3,ds_2)))
        c1 = self.main3(self.conv21(upsample_add(2*c2,ds_1)))
        c0 = self.main4(self.conv10(upsample_add(4*c1,ds_0)))
        f3 = self.conv14(c0)
        f5 = self.conv11(c0)
        f7 = self.conv12(c0)
        f=torch.cat((f3,f5,f7),dim=1)
        f=self.relu1(f)
        f=self.conv13(f)
        return f
class focal_loss(nn.Module):
    def __init__(self,factor=1,gama=1.5,eps=1e-8):
        super(focal_loss,self).__init__()
        self.eps = eps
        self.sig1 = nn.Sigmoid()
        self.gama = gama
        self.factor = factor
    def forward(self,x,y):
        p = self.sig1(x)
        N = y.numel()
        M = torch.sum(y)
        indic1 = torch.nonzero(y)
        indic2 = torch.nonzero(1-y)
        positive = p[indic1[:,0],indic1[:,1],indic1[:,2],indic1[:,3],indic1[:,4]]
        negative = (1-p)[indic2[:,0],indic2[:,1],indic2[:,2],indic2[:,3],indic2[:,4]]
        positive_loss = torch.sum(-((1-positive)**self.gama)*((N-M)/M)*torch.log(positive+self.eps))
        negative_loss = torch.sum(-((1-negative)**self.gama)*torch.log(negative+self.eps))
        loss = self.factor*positive_loss + (2-self.factor)*negative_loss
        loss_mean = loss/N
        return loss_mean

class RandomCrop(object):
    def __init__(self,output_size:int,ispadding:bool=True):
        assert isinstance(output_size,int)
        self.output_size = (output_size,output_size,output_size)
        self.ispadding = ispadding
    def __call__(self,*x):
        y=[]
        d,h,w=x[0].shape
        od,oh,ow=self.output_size
        if self.ispadding:
            x=list(x)
            k1=max(od-d,0);pad1 = k1//2;pads1 = (pad1,pad1) if k1 % 2 == 0 else (pad1,pad1+1);
            k2 = max(oh-h, 0);pad2 = k2 // 2;pads2 = (pad2, pad2) if k2 % 2 == 0 else (pad2, pad2 + 1);
            k3 = max(ow-w, 0);pad3 = k3 // 2;pads3 = (pad3, pad3) if k3 % 2 == 0 else (pad3, pad3 + 1);
            for i in range(len(x)):
                x[i] = np.pad(x[i],(pads1,pads2,pads3),mode='constant')
        d, h, w = x[0].shape
        sd = random.randint(0,d-od)
        sh = random.randint(0,h-oh)
        sw = random.randint(0,w-ow)
        for ix in x:
            y.append(ix[sd:sd+od,sh:sh+oh,sw:sw+ow])
        y = tuple(y)
        return y
