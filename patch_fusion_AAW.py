import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
from model import unet
from metric import dice
from functions.patch_concat.patch_concat import generator_patch
##
model = unet(vol_size=((64,64,64)),enc_nf = [16,32,32,32],dec_nf= [32,32,32,32,32,16,16,3])
model.load_weights('weight/weights.hdf5',by_name=True)
## distance weight
##
xx = np.arange(192)
yy = np.arange(160)
zz = np.arange(160)
grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
##
moving_img = nib.load('data/moving_image/moving_image.nii.gz')
moving_image = moving_img.get_data()
affine = moving_img.affine.copy()
moving_mask = nib.load('data/moving_image/moving_mask.nii.gz').get_data()
##
fixed_image = nib.load('data/fixed_image/MNI_T1_1mm.nii.gz').get_data()
fixed_mask = nib.load('data/fixed_image/MNI_T1_1mm_seg.nii.gz').get_data()
##
moving_image_patch, LOC, fixed_image_patch = generator_patch(moving_image, fixed_image)
##
overlap_flow = np.empty((160, 192, 160, 3))
overlap_weight_flow = np.empty((160, 192, 160, 3))
ones  = np.ones((64,64,64,3))

for i in range(len(moving_image_patch)):
	input1 = moving_image_patch[i]
	input2 = fixed_image_patch[i]
	input2 = input2[np.newaxis,...,np.newaxis]
	pred_temp = model.predict([input1, input2])
	pred_flow = pred_temp[1]
	overlap_flow[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
	LOC[i][2].start:LOC[i][2].stop, :] += pred_flow[0, :, :, :,:]
	overlap_weight_flow[LOC[i][0].start:LOC[i][0].stop,LOC[i][1].start:LOC[i][1].stop,
	LOC[i][2].start:LOC[i][2].stop, :] += ones
flow_predict = overlap_flow / overlap_weight_flow
sample = flow_predict + grid
sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
warp_seg = interpn((yy, xx, zz), moving_mask, sample, method='nearest', bounds_error=False, fill_value=0)
warp_image = interpn((yy, xx, zz), moving_image, sample, method='linear', bounds_error=False, fill_value=0)
val = dice(warp_seg, fixed_mask, labels=[1, 2, 3])
val_mean = np.mean(val)
DSC = [[val[0], val[1], val[2], val_mean], ]
print(DSC)


