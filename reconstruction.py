import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
from modules.losses import reconstruction_loss
import numpy as np
import imageio
from sync_batchnorm import DataParallelWithCallback
from torch import nn
from torchvision import models

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += torch.cdist(x_vgg[i], y_vgg[i]).mean()# self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss / len(self.layids)



def generate(generator, appearance_image, kp_appearance, kp_video, is_3d=False):
    out = {'video_prediction': [], 'video_deformed': []}
    for i in range(kp_video['mean'].shape[1]):
        kp_target = {k: v[:, i:(i + 1)] for k, v in kp_video.items()}
        kp_dict_part = {'kp_driving': kp_target, 'kp_source': kp_appearance}
        out_part = generator(appearance_image, **kp_dict_part)
        out['video_prediction'].append(out_part['video_prediction'])
        if not is_3d:
            out['video_deformed'].append(out_part['video_deformed'])

    out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
    if not is_3d:
        out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)

    out['kp_driving'] = kp_video
    out['kp_source'] = kp_appearance
    return out


def reconstruction(config, generator, kp_detector, checkpoint, log_dir, dataset, opt):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_list = []
    vgg_list = []

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    vgg_distance = VGGLoss()

    generator.eval()
    kp_detector.eval()

    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            kp_appearance = kp_detector(x['video'][:, :, :1])
            d = x['video'].shape[2]
            kp_video = cat_dict([kp_detector(x['video'][:, :, i:(i + 1)]) for i in range(d)], dim=1)
            out = generate(generator, appearance_image=x['video'][:, :, :1], kp_appearance=kp_appearance,
                           kp_video=kp_video, is_3d =opt.is_3d)
            x['source'] = x['video'][:, :, :1]

            # Store to .png for evaluation
            out_video_batch = out['video_prediction'].data.cpu().numpy()
            out_video_batch = np.concatenate(np.transpose(out_video_batch, [0, 2, 3, 4, 1])[0], axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * out_video_batch).astype(np.uint8))

            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)

            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video'].cpu(), 1)
            loss_list.append(loss.data.cpu().numpy())
            vgg_value = vgg_distance(out['video_prediction'].squeeze().transpose(0,1).cuda(), x['video'].squeeze().transpose(0,1).cuda())
            vgg_list.append(vgg_value.data.cpu().numpy())
            del x, kp_video, kp_appearance, out, loss, vgg_value

    print("Reconstruction loss: %s" % np.mean(loss_list))
    print("VGG loss: %s" % np.mean(vgg_list))
