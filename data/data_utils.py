import os
from glob import glob
import numpy as np
from utils.correction_utils import correct

'''get four distortion parameters from the image name'''
def get_img_label(file_name):  
    (_, temp_file_name) = os.path.split(file_name)
    (shot_name, _) = os.path.splitext(temp_file_name)
    split_name = shot_name.split('_')
    k1 = float(split_name[0])
    k2 = float(split_name[1])
    k3 = float(split_name[2])
    k4 = float(split_name[3])
    return k1, k2, k3, k4

'''get the attention soft mask for distortion level estimation'''
def get_gaussian_blob(size):
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    x, y = np.meshgrid(x, y)
    z = np.exp(-(x**2 + y**2))
    return 255.*z

'''get the distortion level given the distortion parameters and the pixel's location'''
def get_dl(k1, k2, k3, k4, i, j, center_x=128, center_y=128):
    rd2 = ((i - center_x) ** 2 + (j - center_y) ** 2)
    rd4 = rd2 * rd2
    rd6 = rd4 * rd2
    rd8 = rd4 * rd4
    dl = 1 / (1 + k1 * rd2 + k2 * rd4 + k3 * rd6 + k4 * rd8)
    return dl   

def get_polynomial(x, y, c=128):
    r_2 = (x - c) ** 2 + (y - c) ** 2
    r_4 = r_2 * r_2
    r_6 = r_2 * r_4
    r_8 = r_4 * r_4
    return r_2, r_4, r_6, r_8

'''compute the distortion parameters from the predicted distortion level'''
def dl2dp(dl1, dl2, dl3, dl4, l1=(144, 144), l2=(176, 176), l3=(208, 208), l4=(240, 240)):
    r1_2, r1_4, r1_6, r1_8 = get_polynomial(l1[0], l1[1])
    r2_2, r2_4, r2_6, r2_8 = get_polynomial(l2[0], l2[1])
    r3_2, r3_4, r3_6, r3_8 = get_polynomial(l3[0], l3[1])
    r4_2, r4_4, r4_6, r4_8 = get_polynomial(l4[0], l4[1])
    
    a = [[r1_2, r1_4, r1_6, r1_8], [r2_2, r2_4, r2_6, r2_8], [r3_2, r3_4, r3_6, r3_8], [r4_2, r4_4, r4_6, r4_8]]
    a = np.array(a)
    b = [(1 / dl1) - 1, (1 / dl2) - 1, (1 / dl3) - 1, (1 / dl4) - 1]
    b = np.array(b)

    x = np.linalg.solve(a, b)
    return x

'''evaluate the MDLD metric of a patch (given the predicted distortion parameters and the ground truth)'''
def mdld(k1, k2, k3, k4, K1, K2, K3, K4, pacth_w=128, patch_h=128):
    sum = 0
    for i in range(pacth_w):
        for j in range(patch_h):
            dl1 = get_dl(k1, k2, k3, k4, i, j)
            dl2 = get_dl(K1, K2, K3, K4, i, j)
            sum += abs(dl1 - dl2)
    
    return sum / (pacth_w * patch_h)   

'''evaluate test set'''
def eva_test(model, img_test_num, pacth_t, block_t, dp_t, dl_t):
    sum_mdld = 0
    c1 = 1e-8 
    c2 = 1e-12
    c3 = 1e-16
    c4 = 1e-20

    for i in range(4*img_test_num):
        img = pacth_t[i]
        img = np.expand_dims(img, axis=0)
        b1 = block_t[i, 0]
        b1 = np.expand_dims(b1, axis=0)
        b2 = block_t[i, 1]
        b2 = np.expand_dims(b2, axis=0)
        b3 = block_t[i, 2]
        b3 = np.expand_dims(b3, axis=0)
        b4 = block_t[i, 3]
        b4 = np.expand_dims(b4, axis=0)
        
        dl = dl_t[i]
        dp = dp_t[i]
        
        # predict dl
        dl_predict = model.predict(x=[img, b1, b2, b3, b4])
        dl_predict = dl_predict.reshape(-1)
        dp_predict = dl2dp(dl_predict[0], dl_predict[1], dl_predict[2], dl_predict[3])
        
        dp1 = dp[0] * c1
        dp2 = dp[1] * c2
        dp3 = dp[2] * c3
        dp4 = dp[3] * c4
        sum_mdld += mdld(dp_predict[0], dp_predict[1], dp_predict[2], dp_predict[3], dp1, dp2, dp3, dp4)    
        
    sum_mdld /= 4*img_test_num
    return sum_mdld

'''correct the test set'''
def eva_correct(model, src_img, img_test_num, pacth_t, block_t):
    results = []
    for i in range(img_test_num):
        img = pacth_t[i]
        img = np.expand_dims(img, axis=0)
        b1 = block_t[i,0]
        b1 = np.expand_dims(b1, axis=0)
        b2 = block_t[i,1]
        b2 = np.expand_dims(b2, axis=0)
        b3 = block_t[i,2]
        b3 = np.expand_dims(b3, axis=0)
        b4 = block_t[i,3]
        b4 = np.expand_dims(b4, axis=0)
        
        dl_predict = model.predict(x=[img, b1, b2, b3, b4])
        dl_predict = dl_predict.reshape(-1)
        dp_predict = dl2dp(dl_predict[0], dl_predict[1], dl_predict[2], dl_predict[3])
        #print(dl_predict, dp_predict)
        img_correction = correct(src_img[i], dp_predict)
        results.append(img_correction)

    return results