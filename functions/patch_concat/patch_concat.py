import numpy as np
from functions.patch_concat.single_data_gen import vols_generator_patch
def EDW(x):
	x_exp = np.exp(-np.array(x))
	x_sum = np.sum(x_exp)
	s = x_exp / x_sum
	return s
def generator_patch(moving_image, fixed_image):
	moving_image_patch,moving_image_patch_loc = vols_generator_patch(vol_name = moving_image,  patch_size = [64,64,64],
											  stride_patch = [32,32,32], out = 2, num_images = 80)
	fixed_image_patch = vols_generator_patch(vol_name = fixed_image, patch_size = [64,64,64],
											 stride_patch = [32,32,32], out = 1, num_images = 80)
	moving_image_patch = moving_image_patch[0]
	moving_image_patch_loc = moving_image_patch_loc[0]
	return moving_image_patch, moving_image_patch_loc, fixed_image_patch
def weight_distance(flowx,flowy,flowz,weight):
	distance_weight = EDW(weight)
	predict_valuex = np.sum(np.multiply(np.array(flowx), np.array(distance_weight)))
	predict_valuey = np.sum(np.multiply(np.array(flowy), np.array(distance_weight)))
	predict_valuez = np.sum(np.multiply(np.array(flowz), np.array(distance_weight)))
	predict_value = (predict_valuex,predict_valuey,predict_valuez)
	return predict_value
def concat_weight(flowx,flowy,flowz,weight):
	flow = np.empty((160,192,160,3))
	for i in range(160):
		for j in range(192):
			for k in range(160):
				flow[i,j,k,:] = weight_distance(flowx[i][j][k],flowy[i][j][k],flowz[i][j][k],weight[i][j][k])
	return flow