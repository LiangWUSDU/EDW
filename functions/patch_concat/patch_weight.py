import numpy as np
import scipy.spatial.distance as dist

def get_patch_3D_distance_weight(winodws, mode):
    patch_height = winodws[0]
    patch_width = winodws[1]
    patch_length = winodws[2]
    patch_weight = np.ones((patch_height, patch_width,patch_length))
    center_x = patch_height / 2 - 1
    center_y = patch_width  / 2 - 1
    center_z = patch_length / 2 - 1
    if mode == 0:  ##Euclidean
        for i in range(patch_height):
            for j in range(patch_width):
                for k in range(patch_length):
                    patch_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'euclidean')
    elif mode == 1: ##cityblock
        for i in range(patch_height):
            for j in range(patch_width):
                for k in range(patch_length):
                    patch_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'cityblock')
    elif mode == 2: ##canberra
        for i in range(patch_height):
            for j in range(patch_width):
                for k in range(patch_length):
                    patch_weight[i,j,k] = dist.pdist(np.array([(i,j,k),(center_x,center_y,center_z)]), 'canberra')
    elif mode == 3:  ##minkowski
        for i in range(patch_height):
            for j in range(patch_width):
                for k in range(patch_length):
                    patch_weight[i, j, k] = dist.pdist(np.vstack([(i, j, k), (center_x, center_y, center_z)]), 'minkowski')
    elif mode == 4:  ##chebyshev
        for i in range(patch_height):
            for j in range(patch_width):
                for k in range(patch_length):
                    patch_weight[i, j, k] = dist.pdist(np.vstack([(i, j, k), (center_x, center_y, center_z)]), 'chebyshev')
    return  patch_weight