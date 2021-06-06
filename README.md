# Animating Arbitrary Objects via Deep Motion Transfer
Implementation of 3D version of Monket-Net. Motion transfer based on 3D optical flow.

# Pretrained checkpoint
[download](https://drive.google.com/file/d/1kr6ACs2IOfD7X-znPJoGx9EGMAKlhFAt/view?usp=sharing)

# Train command
python run.py --is_original --is_3d --config config/vox_3d_original64.yaml --mode train 

# Test command
python run.py --is_original --is_3d --config config/vox_3d_original64.yaml --mode reconstruction
python run.py --is_original --is_3d --config config/vox_3d_original64.yaml --mode transfer
python calculate_akd.py
