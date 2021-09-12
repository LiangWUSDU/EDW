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
from functions.patch_concat.patch_concat import generator_patch,concat_weight
from functions.patch_concat.patch_weight import get_patch_3D_distance_weight
##
model = unet(vol_size=((64,64,64)),enc_nf = [16,32,32,32],dec_nf= [32,32,32,32,32,16,16,3])
model.load_weights('weight/weights.hdf5',by_name=True)
## distance weight
flow_weight = get_patch_3D_distance_weight((64, 64, 64), mode=0)  ##mode=[0,1,2,3,4]
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
##bulid 3D list, same size
mask_flow_x     = [[[[] for ix in range(160)] for jx in range(192)] for kx in range(160)]
mask_flow_y     = [[[[] for iy in range(160)] for jy in range(192)] for ky in range(160)]
mask_flow_z     = [[[[] for iz in range(160)] for jz in range(192)] for kz in range(160)]
position_weight = [[[[] for iw in range(160)] for jw in range(192)] for kw in range(160)]
for i in range(len(moving_image_patch)):
	input1 = moving_image_patch[i]
	input2 = fixed_image_patch[i]
	input2 = input2[np.newaxis,...,np.newaxis]
	pred_temp = model.predict([input1, input2])
	pred_flow = np.squeeze(pred_temp[1])
	ki = kj = kk = 0
	for pi in range(LOC[i][0].start, LOC[i][0].stop):
		if ki > 63:
			ki = 0
		for pj in range(LOC[i][1].start, LOC[i][1].stop):
			if kj > 63:
				kj = 0
			for pk in range(LOC[i][2].start, LOC[i][2].stop):
				if kk > 63:
					kk = 0
				mask_flow_x[pi][pj][pk].append(pred_flow[ki, kj, kk, 0])
				mask_flow_y[pi][pj][pk].append(pred_flow[ki, kj, kk, 1])
				mask_flow_z[pi][pj][pk].append(pred_flow[ki, kj, kk, 2])
				position_weight[pi][pj][pk].append(flow_weight[ki, kj, kk])
				kk += 1
			kj += 1
		ki += 1
	del input1,input2,pred_flow,ki, kj, kk
flow_predict = concat_weight(mask_flow_x,mask_flow_y,mask_flow_z, position_weight)
sample = flow_predict + grid
sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
warp_seg = interpn((yy, xx, zz), moving_mask, sample, method='nearest', bounds_error=False, fill_value=0)
warp_image = interpn((yy, xx, zz), moving_image, sample, method='linear', bounds_error=False, fill_value=0)
val = dice(warp_seg, fixed_mask, labels=[1, 2, 3])
val_mean = np.mean(val)
DSC = [[val[0], val[1], val[2], val_mean], ]
print(DSC)


















