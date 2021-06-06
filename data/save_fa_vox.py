from skimage.io import imread, imsave
from skimage.transform import resize
import os
from tqdm import tqdm
import numpy as np
import cv2
import face_alignment
from skimage import io, img_as_float32

in_dir = '/home/nas3_userM/chaeyeonchung/datasets/unzippedIntervalFaces/data/%s/1.6/'
img_size = (256, 256)
out_dir = '/home/nas3_userM/chaeyeonchung/datasets/VoxCeleb'
#format = '.jpg'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                  device='cuda')  # 'cuda:'+str(device_ids[0])

no_face_list = []

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for partition in ['train', 'test']:
    par_dir = os.path.join(out_dir, partition, 'kp_coords_2d')
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    celebs = open(partition + '_vox1.txt').read().splitlines()
    for celeb in tqdm(celebs):
        celeb_dir = in_dir % celeb
        for video_dir in os.listdir(celeb_dir):
            for part_dir in os.listdir(os.path.join(celeb_dir, video_dir)):
                result_name = celeb + "-" + video_dir + "-" + part_dir
                part_dir = os.path.join(celeb_dir, video_dir, part_dir)
                total_pred = []
                images = os.listdir(part_dir)
                for img_name in sorted(images):
                    image = io.imread(os.path.join(part_dir, img_name))
                    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    image = img_as_float32(image)
                    source = np.array(image, dtype='float32')
                    # print(img_dir)
                    pred = fa.get_landmarks(source * 255)  # source 256 256, 3 -> 0.0 ~ 1.0
                    if pred is not None:
                        total_pred.append(pred[0].astype('int32'))  # 68 2
                    else:
                        break
                if len(total_pred) > 100 or len(total_pred) < 4:
                    print ("Warning sequence of len - %s" % len(total_pred))

                if len(total_pred) < len(images):
                    print("No Face Alert!", len(no_face_list))
                    no_face_list.append(result_name+'.jpg')
                    continue
                else:
                    np.save(os.path.join(par_dir, result_name), total_pred)

print(len(no_face_list))
np.save('no_face_list_vox_2d', no_face_list)