import os
from glob import glob
import cv2
import random
import numpy as np
from data.data_utils import *

'''normal loader'''
def get_img(num, path, patch_w, patch_h, block_w, block_h):
    X_full = np.zeros((4*num, patch_w, patch_h, 3))  # four patches
    blocks = np.zeros((4*num, 4, block_w, block_h, 4)) # four blocks of each patch
    K_full = np.zeros((4*num, 4)) # distortion parameters
    DL_full = np.zeros((4*num, 4)) # distortion levels
    loc_list = glob(path)
    
    patch_loc = [0, 128, 256]
    block_loc_x = [0, 32, 64, 96]
    block_loc_y = [32, 64, 96, 128]
    cof = [1e-8, 1e-12, 1e-16, 1e-20]
    x_loc = [144, 176, 208, 240]
    y_loc = [144, 176, 208, 240]
    
    gausian_blob = get_gaussian_blob(block_w)
    gausian_blob = np.expand_dims(gausian_blob, axis=-1)
    
    for i in range(num):
        img_file_location = loc_list[i]
        img = cv2.imread(img_file_location)
        
        '''devide the image into four patches'''
        img1 = img[patch_loc[0]:patch_loc[1], patch_loc[0]:patch_loc[1]] # top left
        img2 = img[patch_loc[1]:patch_loc[2], patch_loc[0]:patch_loc[1]] # top right
        img3 = img[patch_loc[0]:patch_loc[1], patch_loc[1]:patch_loc[2]] # bottom left
        img4 = img[patch_loc[1]:patch_loc[2], patch_loc[1]:patch_loc[2]] # bottom right
        img1 = cv2.flip(img1, -1, dst = None) 
        img2 = cv2.flip(img2, 1, dst = None) 
        img3 = cv2.flip(img3, 0, dst = None)
        patches = [img1, img2, img3, img4]  
        
        '''parsing the distortion parameters from the file name'''
        k1, k2, k3, k4 = get_img_label(img_file_location)
        k_full_arr = [(k1, k2, k3, k4)]
        k1_c, k2_c, k3_c, k4_c = k1*cof[0], k2*cof[1], k3*cof[2], k4*cof[3]
        
        '''compute the distortion level for each block (given its center)'''
        dl_arr = [0, 0, 0, 0]
        for n in range (4):
            x_c = x_loc[n]
            y_c = y_loc[n]
            dl_arr[n] = get_dl(k1_c, k2_c, k3_c, k4_c, x_c, y_c)
            
        dl_full_arr = [(dl_arr[0], dl_arr[1], dl_arr[2], dl_arr[3])]
        
        '''feel all data into arrays'''
        for j in range(4):
            X_full[int(4*i+j), :, :] = patches[j]
            K_full[int(4*i+j), :] = np.array(k_full_arr).reshape(-1)
            DL_full[int(4*i+j), :] = np.array(dl_full_arr).reshape(-1)
            
            '''devide one patch into four blocks and concat the gaussian blob'''
            for k in range(4):
                block = patches[j][block_loc_x[k]:block_loc_y[k], block_loc_x[k]:block_loc_y[k]]
                blocks[int(4*i+j), k, :, :] = np.concatenate((block, gausian_blob), axis=-1)
    
    '''normalize images'''
    X_full_normal = X_full.astype('float32')
    X_full_normal = (X_full_normal-127.5)/127.5
    
    blocks = blocks.astype('float32')
    blocks = (blocks-127.5)/127.5
    
    return X_full_normal, blocks, K_full, DL_full

'''training loader'''
def load_batch(train_num, batch_size, path, patch_w, patch_h, block_w, block_h):
    X_full = np.zeros((batch_size, patch_w, patch_h, 3))
    blocks = np.zeros((batch_size, 4, block_w, block_h, 4))
    K_full = np.zeros((batch_size, 4))
    DL_full = np.zeros((batch_size, 4))
    loc_list = glob(path)
    
    n_batches = int(train_num/batch_size)
    
    patch_loc = [0, 128, 256]
    block_loc_x = [0, 32, 64, 96]
    block_loc_y = [32, 64, 96, 128]
    cof = [1e-8, 1e-12, 1e-16, 1e-20]
    x_loc = [144, 176, 208, 240]
    y_loc = [144, 176, 208, 240]
    
    gausian_blob = get_gaussian_blob(block_w)
    gausian_blob = np.expand_dims(gausian_blob, axis=-1)
    
    for i in range(n_batches):
        for ii in range(batch_size):
            index = i*batch_size+ii
            img_file_location = loc_list[index]
            img = cv2.imread(img_file_location)
            
            img1 = img[patch_loc[0]:patch_loc[1], patch_loc[0]:patch_loc[1]] # top left
            img2 = img[patch_loc[1]:patch_loc[2], patch_loc[0]:patch_loc[1]] # top right
            img3 = img[patch_loc[0]:patch_loc[1], patch_loc[1]:patch_loc[2]] # bottom left
            img4 = img[patch_loc[1]:patch_loc[2], patch_loc[1]:patch_loc[2]] # bottom right
            img1 = cv2.flip(img1, -1, dst = None) 
            img2 = cv2.flip(img2, 1, dst = None) 
            img3 = cv2.flip(img3, 0, dst = None)
            patches = [img1, img2, img3, img4]
            
            k1, k2, k3, k4 = get_img_label(img_file_location)
            k_full_arr = [(k1, k2, k3, k4)]
            k1_c, k2_c, k3_c, k4_c = k1*cof[0], k2*cof[1], k3*cof[2], k4*cof[3]
            
            dl_arr = [0, 0, 0, 0]
            for n in range (4):
                x_c = x_loc[n]
                y_c = y_loc[n]
                dl_arr[n] = get_dl(k1_c, k2_c, k3_c, k4_c, x_c, y_c)
                
            dl_full_arr = [(dl_arr[0], dl_arr[1], dl_arr[2], dl_arr[3])]
            
            '''randomly select a patch of an image for training'''
            random_index = random.randint(0, 3)
            X_full[ii, :, :] = patches[random_index]
            K_full[ii, :] = np.array(k_full_arr).reshape(-1)
            DL_full[ii, :] = np.array(dl_full_arr).reshape(-1)
                
            for k in range(4):
                block = patches[random_index][block_loc_x[k]:block_loc_y[k], block_loc_x[k]:block_loc_y[k]]
                blocks[ii, k, :, :] = np.concatenate((block, gausian_blob), axis=-1)
    
        index_shuffle = [i for i in range(X_full.shape[0])]
        random.shuffle(index_shuffle)
        X_full = X_full[index_shuffle]
        K_full = K_full[index_shuffle]
        blocks = blocks[index_shuffle]
        DL_full = DL_full[index_shuffle]
        
        X_full_normal = X_full.astype('float32')
        X_full_normal = (X_full_normal-127.5)/127.5
        
        blocks = blocks.astype('float32')
        blocks = (blocks-127.5)/127.5
        
        yield X_full_normal, blocks, K_full, DL_full

'''inference loader from scratch (only choose one patch in an image)'''
def get_test_img(num, path, img_w, img_h, patch_w, patch_h, block_w, block_h, patch_index=0):
    src = np.zeros((num, img_w, img_h, 3))
    X_full = np.zeros((num, patch_w, patch_h, 3))
    blocks = np.zeros((num, 4, block_w, block_h, 4))
    loc_list = glob(path)
    file_name = []
    
    patch_loc = [0, 128, 256]
    block_loc_x = [0, 32, 64, 96]
    block_loc_y = [32, 64, 96, 128]
    
    gausian_blob = get_gaussian_blob(block_w)
    gausian_blob = np.expand_dims(gausian_blob, axis=-1)
    
    for i in range(num):
        img_file_location = loc_list[i]
        img = cv2.imread(img_file_location)
        src[i] = img
        _, name = os.path.split(img_file_location)
        file_name.append(name)
        '''devide the image into four patches'''
        img1 = img[patch_loc[0]:patch_loc[1], patch_loc[0]:patch_loc[1]] # top left
        img2 = img[patch_loc[1]:patch_loc[2], patch_loc[0]:patch_loc[1]] # top right
        img3 = img[patch_loc[0]:patch_loc[1], patch_loc[1]:patch_loc[2]] # bottom left
        img4 = img[patch_loc[1]:patch_loc[2], patch_loc[1]:patch_loc[2]] # bottom right
        img1 = cv2.flip(img1, -1, dst = None) 
        img2 = cv2.flip(img2, 1, dst = None) 
        img3 = cv2.flip(img3, 0, dst = None)
        patches = [img1, img2, img3, img4]
        
        if(patch_index!=4):
            X_full[i] = patches[patch_index]
            for k in range(4):
                block = patches[patch_index][block_loc_x[k]:block_loc_y[k], block_loc_x[k]:block_loc_y[k]]
                blocks[i, k, :, :] = np.concatenate((block, gausian_blob), axis=-1)
        else:
            random_index = random.randint(0, 3)
            X_full[i] = patches[random_index]
            for k in range(4):
                block = patches[random_index][block_loc_x[k]:block_loc_y[k], block_loc_x[k]:block_loc_y[k]]
                blocks[i, k, :, :] = np.concatenate((block, gausian_blob), axis=-1)
    
    '''normalize images'''
    X_full_normal = X_full.astype('float32')
    X_full_normal = (X_full_normal-127.5)/127.5
    
    blocks = blocks.astype('float32')
    blocks = (blocks-127.5)/127.5
    
    return src, X_full_normal, blocks, file_name