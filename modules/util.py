from torch import nn

import torch.nn.functional as F
import torch

import numpy as np
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


def compute_image_gradient(image, padding=0):
    bs, c, h, w = image.shape

    sobel_x = torch.from_numpy(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])).type(image.type())
    filter = sobel_x.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(image, filter, groups=c, padding=padding)
    grad_x = grad_x

    sobel_y = torch.from_numpy(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])).type(image.type())
    filter = sobel_y.unsqueeze(0).repeat(c, 1, 1, 1)
    grad_y = F.conv2d(image, filter, groups=c, padding=padding)
    grad_y = grad_y

    return torch.cat([grad_x, grad_y], dim=1)


def make_coordinate_grid(spatial_size, type): # 256,256 -> 256, 256, 2 (-1, 1)
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def make_coordinate_grid_3d(spatial_size, type): # 256 256 256 -> 256 256 256 3
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w, d = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    zz = z.view(-1, 1, 1).repeat(1, h, w)
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


class ResBlock3D(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = x
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, is_3d = False):
        super(UpBlock3D, self).__init__()
        self.scale_factor = (2 ,2 ,2) if is_3d else (1, 2, 2)
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale_factor)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock3D(nn.Module):
    """
    Simple block with group convolution.
    """

    def __init__(self, in_features, out_features, groups=None, kernel_size=3, padding=1):
        super(SameBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256, temporal=False, is_3d = False, conv1_out = 4096):
        super(Encoder, self).__init__()
        self.is_3d = is_3d

        down_blocks = []

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

        if self.is_3d:
            self.conv1_first = nn.Conv3d(in_channels=in_features, out_channels=in_features * 64,
                                         kernel_size=(1, 1, 1))  # 128 256
            conv1_blocks = []
            in_feat = 64
            for i in range(num_blocks-1): #num_blocks
                conv1_blocks.append(nn.Conv3d(in_channels=in_feat * (2 ** i) , out_channels= conv1_out, kernel_size=(1, 1, 1)))
            conv1_blocks.append(nn.Conv3d(in_channels=512, out_channels=int(conv1_out/2), kernel_size=(1, 1, 1))) # X
            self.conv1_blocks = nn.ModuleList(conv1_blocks)

        # torch.Size([4, 3, 2, 256, 256])
        # torch.Size([4, 64, 2, 128, 128])
        # torch.Size([4, 128, 2, 64, 64])
        # torch.Size([4, 256, 2, 32, 32])
        # torch.Size([4, 512, 2, 16, 16])
        # torch.Size([4, 1024, 2, 8, 8])

    def forward(self, x):
        x_shape = x.shape

        if self.is_3d:
            out = x
            outs = [self.conv1_first(out).reshape(x_shape[0], x_shape[1], -1, x_shape[2], x_shape[3], x_shape[4]).permute(0, 3, 1, 2, 4, 5).reshape(-1, x_shape[1], x_shape[3], x_shape[3], x_shape[4])]

            for down_block, conv1 in zip( self.down_blocks, self.conv1_blocks):

                out = down_block(out)
                out2 = conv1(out)  # conv1 , reshape
                out_shape = out.shape
                outs.append(out2.reshape(out_shape[0], out_shape[1], -1, out_shape[2], out_shape[3], out_shape[4]).permute(0, 3, 1, 2, 4, 5).reshape(-1, out_shape[1], out_shape[3], out_shape[3], out_shape[4]))  # B, 2, 1024, 8, 8, 8
                # torch.Size([2, 64, 1, 64, 64]) torch.Size([2, 1024, 1, 64, 64])
        else:
            outs = [x]
            for down_block in self.down_blocks:
                outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False,
                 additional_features_for_block=0, use_last_conv=True, is_3d = False, to_2d = False, is_original = False):
        super(Decoder, self).__init__()
        self.to_2d = to_2d
        self.is_3d = is_3d
        self.is_original = is_original

        kernel_size = (3, 3, 3) if temporal else (1, 3, 3)
        padding = (1, 1, 1) if temporal else (0, 1, 1)

        if self.is_3d:
            if self.to_2d:
                in_ch = 2880 if self.is_original else 5760 # 5760 #13184
                self.conv_to_2d1 = nn.Conv3d(in_channels=in_ch, out_channels=int(in_ch/16), kernel_size=kernel_size, padding=padding)
                self.conv_to_2d2 = nn.Conv3d(in_channels=int(in_ch/16), out_channels=45, kernel_size=kernel_size, padding=padding)
                self.conv_to_2d3 = BatchNorm3d(45, affine=True)
                self.conv_to_2d4 = nn.LeakyReLU()
                self.conv_to_2d5 = nn.Conv3d(in_channels=45, out_channels=45, kernel_size=(1, 1, 1))

            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)



        up_blocks = []

        for i in range(num_blocks)[::-1]:
            up_blocks.append(UpBlock3D((1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (
                2 ** (i + 1))) + additional_features_for_block,
                                       min(max_features, block_expansion * (2 ** i)),
                                       kernel_size=kernel_size, padding=padding, is_3d = is_3d))

        self.up_blocks = nn.ModuleList(up_blocks)
        if use_last_conv:
            self.conv = nn.Conv3d(in_channels=block_expansion + in_features + additional_features_for_block,
                                  out_channels=out_features, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = None



    def forward(self, x):
        out = x.pop() # out 0 이 input, out 1 이 concat 대상
        # torch.Size([4, 3, 2, 256, 256])
        # torch.Size([4, 64, 2, 128, 128])
        # torch.Size([4, 128, 2, 64, 64])
        # torch.Size([4, 256, 2, 32, 32])
        # torch.Size([4, 512, 2, 16, 16])
        # torch.Size([4, 1024, 2, 8, 8])
        # B, 2, 1024, 8, 8, 8

        # torch.Size([2, 3, 128, 128, 128])
        # torch.Size([4, 64, 64, 64, 64])
        # torch.Size([4, 128, 32, 32, 32])
        # torch.Size([4, 256, 16, 16, 16])
        # torch.Size([4, 512, 8, 8, 8])
        # torch.Size([4, 1024, 4, 4, 4])

        # torch.Size([1, 13, 128, 128, 128])
        # torch.Size([1, 74, 64, 64, 64])
        # torch.Size([1, 138, 32, 32, 32])
        # torch.Size([1, 266, 16, 16, 16])
        # torch.Size([1, 522, 8, 8, 8])
        # torch.Size([1, 1034, 4, 4, 4])

        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)

        # [4, 35, 2, 256, 256] 8 35 256 256 256

        if self.conv is not None:
            out =  self.conv(out)
        if self.is_3d:
            out_shape = out.shape
            if self.to_2d:
                out = out.reshape(out_shape[0],-1, 1, out_shape[-2],out_shape[-1])
                out = self.conv_to_2d1(out)
                out = self.conv_to_2d2(out)
                out = self.conv_to_2d3(out)
                out = self.conv_to_2d4(out)
                out = self.conv_to_2d5(out)
            else:
                ch = int(out_shape[0]/2)
                out = out.reshape(ch if ch else 1 , out_shape[1], -1, out_shape[2], out_shape[3], out_shape[4]) # 8 10 256 256 256 -> 4 10 2 256 256 256

        return out # [4, 10, 2, 256, 256] 4 10 2 256 256 256


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256, temporal=False, is_3d = False, conv1_out = 4096):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features, temporal=temporal, is_3d = is_3d, conv1_out = conv1_out)
        self.decoder = Decoder(block_expansion, in_features, out_features, num_blocks, max_features, temporal=temporal, is_3d = is_3d)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def matrix_inverse(batch_of_matrix, eps=0):
    if eps != 0:
        init_shape = batch_of_matrix.shape
        a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
        b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
        c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
        d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

        det = a * d - b * c
        out = torch.cat([d, -b, -c, a], dim=-1)
        eps = torch.tensor(eps).type(out.type())
        out /= det.max(eps)

        return out.view(init_shape)
    else:
        b_mat = batch_of_matrix
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        b_inv, _ = torch.solve(eye, b_mat) #gesv
        return b_inv


def matrix_det(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    det = a * d - b * c
    return det


def matrix_trace(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    return a + d


def smallest_singular(batch_of_matrix):
    a = batch_of_matrix[..., 0, 0].unsqueeze(-1)
    b = batch_of_matrix[..., 0, 1].unsqueeze(-1)
    c = batch_of_matrix[..., 1, 0].unsqueeze(-1)
    d = batch_of_matrix[..., 1, 1].unsqueeze(-1)

    s1 = a ** 2 + b ** 2 + c ** 2 + d ** 2
    s2 = (a ** 2 + b ** 2 - c ** 2 - d ** 2) ** 2
    s2 = torch.sqrt(s2 + 4 * (a * c + b * d) ** 2)

    norm = torch.sqrt((s1 - s2) / 2)
    return norm
