from torch import nn
import torch
import torch.nn.functional as F
from modules.util import make_coordinate_grid, make_coordinate_grid_3d
from modules.keypoint_detector import kp2gaussian, kp2gaussian_3d

class MovementEmbeddingModule(nn.Module):
    """
    Produce a keypoint representation that will be further used by other modules
    """

    def __init__(self, num_kp, kp_variance, num_channels, use_deformed_source_image=False, use_difference=False,
                 use_heatmap=True, add_bg_feature_map=False, heatmap_type='gaussian', norm_const='sum', scale_factor=1, is_3d = False, is_original = False, debug = None):
        super(MovementEmbeddingModule, self).__init__()

        assert heatmap_type in ['gaussian', 'difference']

        assert ((int(use_heatmap) + int(use_deformed_source_image) + int(use_difference)) >= 1)
        self.debug= debug
        self.is_3d = is_3d
        self.is_original = is_original
        self.out_channels = (1 * use_heatmap + 2 * use_difference + num_channels * use_deformed_source_image) * (
            num_kp + add_bg_feature_map)

        self.kp_variance = kp_variance
        self.heatmap_type = heatmap_type
        self.use_difference = use_difference
        self.use_deformed_source_image = use_deformed_source_image
        self.use_heatmap = use_heatmap
        self.add_bg_feature_map = add_bg_feature_map
        self.norm_const = norm_const
        self.scale_factor = scale_factor
        self.kp2gaussian = kp2gaussian_3d if self.is_3d else kp2gaussian

    def normalize_heatmap(self, heatmap):
        if self.norm_const == "sum":
            heatmap_shape = heatmap.shape
            heatmap = heatmap.view(heatmap_shape[0], heatmap_shape[1], heatmap_shape[2], -1)
            heatmap = heatmap / heatmap.sum(dim=3, keepdim=True)
            return heatmap.view(*heatmap_shape)
        else:
            return heatmap / self.norm_const
        

    def forward(self, source_image, kp_driving, kp_source, source_image_3d=None):
        if self.scale_factor != 1:
            source_image = F.interpolate(source_image, scale_factor=(1, self.scale_factor, self.scale_factor))

        spatial_size = source_image.shape[3:]

        bs, _, _, h, w = source_image.shape
        _, d, num_kp, _ = kp_driving['mean'].shape

        # gt_heatmap = kp_driving['heatmap32'] if (h == 32 or h == 16) else kp_driving['heatmap']
        inputs = []
        if self.use_heatmap:
            heatmap = self.normalize_heatmap(
                self.kp2gaussian(kp_driving, spatial_size= (h, h, h) if self.is_3d else spatial_size , kp_variance=self.kp_variance))
            heatmap_appearance = self.normalize_heatmap(
                self.kp2gaussian(kp_source, spatial_size= (h, h, h) if self.is_3d else spatial_size, kp_variance=self.kp_variance))
            heatmap = heatmap - heatmap_appearance
            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, h, h, h).type(heatmap.type()) if self.is_3d else torch.zeros(bs, d, 1, h, w).type(heatmap.type())
                heatmap = torch.cat([zeros, heatmap], dim=2)
            heatmap = heatmap.unsqueeze(3)
            inputs.append(heatmap) # [2, 1, 11, 1, 128, 128]

        num_kp += self.add_bg_feature_map
        if self.use_difference or self.use_deformed_source_image:
            kp_video_diff = kp_source['mean'] - kp_driving['mean']
            if self.add_bg_feature_map:
                zeros = torch.zeros(bs, d, 1, 3).type(kp_video_diff.type()) if self.is_3d else torch.zeros(bs, d, 1, 2).type(kp_video_diff.type())
                kp_video_diff = torch.cat([zeros, kp_video_diff], dim=2)
            kp_video_diff = kp_video_diff.view((bs, d, num_kp, 3, 1, 1, 1)).repeat(1, 1, 1, 1, h, h, h) if self.is_3d else kp_video_diff.view((bs, d, num_kp, 2, 1, 1)).repeat(1, 1, 1, 1, h, w)

        if self.use_difference:
            inputs.append(kp_video_diff)

        if self.use_deformed_source_image and not self.is_3d:
            if self.is_3d: #[8, 3, 64, 64, 64]
                appearance_repeat = source_image_3d.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)  # [8, 1, 11, 3, 64, 64, 64]
                appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, h, h)  # [22, 3, 128, 128] -> [88, 3, 64, 64, 64]
                deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, h, h)).permute(0, 2, 3, 4, 1)  # 88, 64, 64, 64, 3
                coordinate_grid = make_coordinate_grid_3d((h, h, h), type=deformation_approx.type())
                coordinate_grid = coordinate_grid.view(1, h, h, h, 3)  # [1 64, 64, 64, 3]
            else:
                appearance_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, d, num_kp, 1, 1, 1, 1)  # [8, 1, 11, 3, 1, 64, 64]
                appearance_repeat = appearance_repeat.view(bs * d * num_kp, -1, h, w)  # [22, 3, 128, 128] -> [88, 3, 64, 64]
                deformation_approx = kp_video_diff.view((bs * d * num_kp, -1, h, w)).permute(0, 2, 3, 1) # [22, 2, 128, 128] -> [22, 128, 128, 2]
                coordinate_grid = make_coordinate_grid((h, w), type=deformation_approx.type())
                coordinate_grid = coordinate_grid.view(1, h, w, 2)  # [1 128, 128, 2]

            deformation_approx = coordinate_grid + deformation_approx # [22, 128, 128, 2] 88, 64, 64, 64, 3

            appearance_approx_deform = F.grid_sample(appearance_repeat, deformation_approx)
            if self.is_3d:
                appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, h, h)) # [2, 1, 11, 3, 128, 128]
            else:
                appearance_approx_deform = appearance_approx_deform.view((bs, d, num_kp, -1, h, w)) # [2, 1, 11, 3, 128, 128]
            inputs.append(appearance_approx_deform)
        movement_encoding = torch.cat(inputs, dim=3) # [2, 1, 11, 1, 128, 128] [2, 1, 11, 3, 128, 128] ->  [2, 1, 11, 1, 128, 128, 128] [2, 1, 11, 3, 128, 128, 128]
        movement_encoding = movement_encoding.view(bs, d, -1, h, h, h) if self.is_3d else movement_encoding.view(bs, d, -1, h, w)  # [2, 1, 11, 1, 128, 128, 128]
        return movement_encoding.permute(0, 2, 1, 3, 4, 5) if self.is_3d else movement_encoding.permute(0, 2, 1, 3, 4)
