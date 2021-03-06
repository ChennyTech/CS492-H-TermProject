import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from augmentation import AllAugmentationTransform, VideoToTensor
from utils_data import *
import glob

def flip(img):
    img = cv2.flip(img[0], 1)
    return np.expand_dims(img, axis=0)

def resize_crop(img, x, y, img_size=None):
    if img_size is not None:
        img = cv2.resize(img[0], dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, axis=0)
    return img[:, y: y + 256, x: x + 256, :]

def create_gaussian_heatmap_3d(keypoints, img_size): #1 68 2, 256
    # keypoints shape: (batch, #ofkeypoint, 2)
    heatmaps = np.zeros((keypoints.shape[0], keypoints.shape[1], img_size, img_size, img_size), dtype=np.float32)
    for i in range(keypoints.shape[0]):
        for p in range(keypoints.shape[1]):
            heatmaps[i,p] = draw_gaussian_3d(heatmaps[i,p], ( keypoints[i,p] / 256 ) * img_size , 5)
    #return torch.tensor(heatmaps)
    return heatmaps.transpose(0,2,3,4,1)

def create_gaussian_heatmap(keypoints, img_size): #1 68 2, 256
    # keypoints shape: (batch, #ofkeypoint, 2)
    heatmaps = np.zeros((keypoints.shape[0], keypoints.shape[1], img_size, img_size), dtype=np.float32)
    for i in range(keypoints.shape[0]):
        for p in range(keypoints.shape[1]):
            heatmaps[i,p] = draw_gaussian(heatmaps[i,p], ( keypoints[i,p] / 256 ) * img_size, 5)
    #return torch.tensor(heatmaps)
    return heatmaps.transpose(0,2,3,1)

def read_video(name, image_shape):
    if name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)

    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class VoxCeleb(Dataset):
    def __init__(self, opt, augmentation_params, image_shape = (256,256, 3), pairs_list= None):
        self.opt = opt
        self.pairs_list = pairs_list
        if self.opt.mode == 'train':
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = VideoToTensor()

        self.image_shape = tuple(image_shape)

        root_dir = '/home/nas3_userM/chaeyeonchung/datasets/VoxCeleb/'

        self.data_dir = os.path.join(root_dir, 'train' if self.opt.mode == 'train' else 'test')

        self.images = os.listdir(self.data_dir)

        if self.opt.mode == 'train':
            self.images = self.images[:int(len(self.images) / 2)]
            self.images.remove('kp_coords_2d')
            self.images.remove('kp_coords_3d')
        self.len = len(self.images)
        print("Data len: ", self.len)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = os.path.join(self.data_dir, self.images[index])
        video_array = read_video(img_name, image_shape=(256,256, 3))

        out = self.transform(video_array)

        if self.opt.mode == 'reconstruction' or self.opt.mode == 'transfer':
            pass
            #resize ??????

        out['name'] = os.path.basename(img_name)
        return out


class FramesDataset(Dataset):
    """Dataset of videos, videos can be represented as an image of concatenated frames, or in '.mp4','.gif' format"""

    def __init__(self, root_dir, augmentation_params, image_shape=(64, 64, 3), is_train=True,
                 random_seed=0, pairs_list=None, transform=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.pairs_list = pairs_list

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

        if transform is None:
            if is_train:
                self.transform = AllAugmentationTransform(**augmentation_params)
            else:
                self.transform = VideoToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])

        video_array = read_video(img_name, image_shape=self.image_shape)

        out = self.transform(video_array)
        # add names
        out['name'] = os.path.basename(img_name)

        return out


class PairedDataset(Dataset):
    """
    Dataset of pairs for transfer.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            images = self.initial_dataset.images
            name_to_index = {name: index for index, name in enumerate(images)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(images), pairs['driving'].isin(images))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
