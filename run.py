import matplotlib

matplotlib.use('Agg')

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset, VoxCeleb

from modules.generator import MotionTransferGenerator
from modules.discriminator import Discriminator
from modules.keypoint_detector import KPDetector

from train import train
from reconstruction import reconstruction
from transfer import transfer
from prediction import prediction

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "test", "reconstruction", "transfer", "prediction"])
    parser.add_argument("--is_original", action='store_true', help='original or keypoint gt')
    parser.add_argument("--is_3d", action='store_true', help='2d or 3d')
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default='/home/nas3_userM/chaeyeonchung/monkey-net_3d/log/vox_2d_original 04-06-21 10:55:11/00000019-checkpoint.pth.tar', help="path to checkpoint to restore")
    #parser.add_argument("--checkpoint", default='/home/nas3_userM/chaeyeonchung/monkey-net_3d/log/vox_3d_original64 04-06-21 09:33:34/00000019-checkpoint.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    #parser.add_argument("--num_of_kp", default="10", type=int, help="Number of keypoints")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)
        blocks_discriminator = config['model_params']['discriminator_params']['num_blocks']
        assert len(config['train_params']['loss_weights']['reconstruction']) == blocks_discriminator + 1

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    generator = MotionTransferGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'], opt = opt)
    generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = Discriminator(**config['model_params']['discriminator_params'],
                                  **config['model_params']['common_params'])
    discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'], opt = opt)
    kp_detector.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_detector)

    dataset = VoxCeleb(opt = opt, **config['dataset_params'])
    #dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids, opt)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt)
    elif opt.mode == 'transfer':
        print("Transfer...")
        transfer(config, generator, kp_detector, opt.checkpoint, log_dir, dataset, opt)
    elif opt.mode == "prediction":
        print("Prediction...")
        prediction(config, generator, kp_detector, opt.checkpoint, log_dir, opt)

#