from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, SameBlock3D, make_coordinate_grid_3d
from modules.movement_embedding import MovementEmbeddingModule

class DenseMotionModule(nn.Module):
    """
    Module that predicting a dense optical flow only from the displacement of a keypoints
    and the appearance of the first frame
    """

    def __init__(self, block_expansion, num_blocks, max_features, mask_embedding_params, num_kp,
                 num_channels, kp_variance, use_correction, use_mask, bg_init=2, num_group_blocks=0, scale_factor=1, is_3d = False, is_original = False):
        super(DenseMotionModule, self).__init__()
        self.is_3d = is_3d
        self.mask_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance, num_channels=num_channels,
                                                      add_bg_feature_map=True, **mask_embedding_params, is_3d = is_3d, is_original = is_original)
        self.difference_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                            num_channels=num_channels,
                                                            add_bg_feature_map=True, use_difference=True,
                                                            use_heatmap=False, use_deformed_source_image=False, is_3d = is_3d, is_original = is_original)
        if is_3d:
            self.hourglass = nn.Conv3d(11, (num_kp + 1) * use_mask + 3 * use_correction, kernel_size=(1, 1, 1)) #self.mask_embedding.out_channels

        else:
            group_blocks = []
            for i in range(num_group_blocks):
                group_blocks.append(SameBlock3D(self.mask_embedding.out_channels, self.mask_embedding.out_channels,
                                                groups=num_kp + 1, kernel_size=(1, 1, 1), padding=(0, 0, 0)))
            self.group_blocks = nn.ModuleList(group_blocks)

            self.hourglass = Hourglass(block_expansion=block_expansion, in_features=self.mask_embedding.out_channels,
                                       out_features=(num_kp + 1) * use_mask + 2 * use_correction,
                                       max_features=max_features, num_blocks=num_blocks, is_3d=is_3d, conv1_out=1024)
            self.hourglass.decoder.conv.weight.data.zero_()
            bias_init = ([bg_init] + [0] * num_kp) * use_mask + [0, 0] * use_correction
            self.hourglass.decoder.conv.bias.data.copy_(torch.tensor(bias_init, dtype=torch.float))

        self.num_kp = num_kp
        self.use_correction = use_correction
        self.use_mask = use_mask
        self.scale_factor = scale_factor

    def forward(self, source_image, kp_driving, kp_source, source_image_3d=None):
        prediction = self.mask_embedding(source_image, kp_driving, kp_source, source_image_3d=source_image_3d)

        if self.is_3d:
            prediction = prediction.squeeze(2)
        else:
            for block in self.group_blocks:
                prediction = block(prediction)
                prediction = F.leaky_relu(prediction, 0.2)

        prediction = self.hourglass(prediction)

        bs, _, d, h, w = prediction.shape
        if self.use_mask:
            mask = prediction[:, :(self.num_kp + 1)]
            mask = F.softmax(mask, dim=1)
            mask = mask.unsqueeze(2)
            difference_embedding = self.difference_embedding(source_image, kp_driving, kp_source)
            difference_embedding = difference_embedding.view(bs, self.num_kp + 1, 3 if self.is_3d else 2, d, h, w)
            deformations_relative = (difference_embedding * mask).sum(dim=1)
        else:
            deformations_relative = 0

        if self.use_correction:
            correction = prediction[:, -3:] if self.is_3d else prediction[:, -2:]
        else:
            correction = 0

        deformations_relative = deformations_relative + correction # [B, 3, 128, 128, 128]
        deformations_relative = deformations_relative.permute(0, 2, 3, 4, 1) # [B, 2, 1, 128, 128] -> ([B, 1, 128, 128, 2] // [B, 128, 128, 128, 3]

        if self.is_3d:
            coordinate_grid = make_coordinate_grid_3d((h, w, w), type=deformations_relative.type())
            coordinate_grid = coordinate_grid.unsqueeze(0)
        else:
            coordinate_grid = make_coordinate_grid((h, w), type=deformations_relative.type())
            coordinate_grid = coordinate_grid.view(1, 1, h, w, 2) # [1, 1, 128, 128, 2] [1, 128, 128, 128, 3]
        deformation = deformations_relative + coordinate_grid # [B, 1, 128, 128, 2] -> [B, 128, 128, 128, 3]

        if self.is_3d:
            return deformation
        else:
            z_coordinate = torch.zeros(deformation.shape[:-1] + (1,)).type(deformation.type()) # ([1, 1, 128, 128, 1])
            return torch.cat([deformation, z_coordinate], dim=-1) # [B, 1, 128, 128, 2] -> [B, 128, 128, 128, 3]


class IdentityDeformation(nn.Module):
    def forward(self, appearance_frame, kp_video, kp_appearance):
        bs, _, _, h, w = appearance_frame.shape
        _, d, num_kp, _ = kp_video['mean'].shape
        coordinate_grid = make_coordinate_grid((h, w), type=appearance_frame.type())
        coordinate_grid = coordinate_grid.view(1, 1, h, w, 2).repeat(bs, d, 1, 1, 1)

        z_coordinate = torch.zeros(coordinate_grid.shape[:-1] + (1,)).type(coordinate_grid.type())
        return torch.cat([coordinate_grid, z_coordinate], dim=-1)
