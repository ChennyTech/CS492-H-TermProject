import face_alignment
import numpy as np
from tqdm import tqdm
import os
from skimage import io, img_as_float32
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageDraw

#with Image.open(img_dir) as im:
#    im = im.resize((256, 256))
#    draw = ImageDraw.Draw(im)
#    draw.point(list(zip(pred[:,0], pred[:,1])))
#    im.save('sample_keypoints.png')
"""
fig = plt.figure(figsize=(256, 256))
ax = fig.gca(projection = '3d')
ax.scatter(-pred[:,0], -pred[:,2], -pred[:,1])
plt.savefig('sample_fig.svg')
"""

fake_dir = '/home/nas3_userM/chaeyeonchung/monkey-net_3d/log/vox_2d_original 04-06-21 10:55:11/reconstruction/png'
real_dir = '/home/nas3_userM/chaeyeonchung/datasets/VoxCeleb/test'

img_list = os.listdir(fake_dir)


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                  device='cuda')  # 'cuda:'+str(device_ids[0])

kp_dist = 0
cnt = 0
for img_path in tqdm(img_list):
    fake_path = os.path.join(fake_dir, img_path)
    fake_img = io.imread(fake_path)
    fake_img = fake_img.reshape(64, -1, 64,3).transpose(1,0,2,3)

    real_name = img_path.split('.jpg')[0] +'.jpg'
    real_path = os.path.join(real_dir, real_name)
    real_img = io.imread(real_path)
    real_img = real_img.reshape(256, -1, 256, 3).transpose(1, 0, 2, 3)
    tmp_dist = 0
    tmp_cnt = 0
    for i in range(fake_img.shape[0]):
        fake_img1 = cv2.resize(fake_img[i], dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        fake_img1 = img_as_float32(fake_img1)
        fake_img1 = np.array(fake_img1, dtype='float32')
        pred_fake = fa.get_landmarks(fake_img1 * 255)
        kp_fake = pred_fake[0].astype('int32')

        real_img1 = real_img[i]
        real_img1 = img_as_float32(real_img1)
        real_img1 = np.array(real_img1, dtype='float32')
        pred_real = fa.get_landmarks(real_img1 * 255)  # source 256 256, 3 -> 0.0 ~ 1.0
        kp_real = pred_real[0].astype('int32')

        tmp_dist += abs(kp_fake - kp_real).mean() # 0.9
        tmp_cnt += 1

    kp_dist += tmp_dist / tmp_cnt
    cnt += 1

print('key point distance: ', kp_dist/cnt)



