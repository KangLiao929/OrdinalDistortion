import os
from glob import glob
import math
import numpy as np
import cv2

def radial_div_r(dp, i, j, center_x, center_y):
    rd2 = ((i - center_x) ** 2 + (j - center_y) ** 2)
    rd4 = rd2 * rd2
    rd6 = rd4 * rd2
    rd8 = rd4 * rd4
    k = 1 / (1 + dp[0] * rd2 + dp[1] * rd4 + dp[2] * rd6 + dp[3] * rd8)
    u = center_x + (i - center_x) * k
    v = center_y + (j - center_y) * k
    return u, v

def average_color(gt, gt_flag, j, i, dir_x, dir_y):
    p_n = 0
    b = g = r = 0
    for k in range(4):
        ni, nj = i + dir_x[k], j + dir_y[k]
        if gt_flag[nj, ni] != 0:
            b += gt[nj, ni][0]
            g += gt[nj, ni][1]
            r += gt[nj, ni][2]
            p_n += 1
    if p_n > 0:
        gt[j, i] = [int(b / p_n), int(g / p_n), int(r / p_n)]
    gt_flag[j, i] = 1

def fill_blank(gt, gt_flag, w=256, h=256):
    dir_x = [-1, 0, 1, 0]
    dir_y = [0, -1, 0, 1]
    for j in range(1, h - 1):
        for i in range(1, w - 1):
            if i < w - j and gt_flag[j, i] == 0:
                average_color(gt, gt_flag, j, i, dir_x, dir_y)
    for j in range(h - 2, 1, -1):
        for i in range(w - 2, 1, -1):
            if i >= w - j and gt_flag[j, i] == 0:
                average_color(gt, gt_flag, j, i, dir_x, dir_y)
    return gt

def correct(srcImg, dp, w=256, h=256, blank_color=255):
    gt = np.zeros((w, h, 3), np.uint8) 
    gt.fill(blank_color)
    center_x = w / 2
    center_y = h / 2
    crop_low = 3
    crop_up = 251
    start_point = 0
    end_point = 256
    flag_bd = 1.5
    
    '''select effective region'''
    for j in range(h):
        for i in range(w):
            x, y = radial_div_r(dp, i, j, center_x, center_y)
            if abs(x) < flag_bd and abs(y) < flag_bd:
                start_point = j
            if abs(x) < flag_bd and abs(y - h) < flag_bd:
                end_point = j
    
    w_roi = end_point - start_point
    offset = start_point
    factor = w_roi / w
    print(w_roi, offset, factor)

    '''backward warping'''
    gt_flag = np.zeros((w, h), np.uint8)
    for j in range(0, w):
        for i in range(0, h):
            jj = int(j * factor + offset)
            ii = int(i * factor + offset)
            x, y = radial_div_r(dp, ii, jj, center_x, center_y)
            if 0 <= x < w and 0 <= y < h:
                xf = int(np.floor(x))
                xc = int(np.ceil(x))
                yf = int(np.floor(y))
                yc = int(np.ceil(y))
                xf_t = xf - x
                xc_t = x - xc
                yf_t = yf - y
                yc_t = y - yc
                x = xf if xf_t < xc_t else xc
                y = yf if yf_t < yc_t else yc
                xf = max(crop_low, min(xf, crop_up))
                yf = max(crop_low, min(yf, crop_up))
                gt[yf, xf] = srcImg[j, i]
                gt_flag[yf, xf] = 1
    
    gt = fill_blank(gt, gt_flag) # interpolation
    gt = gt[crop_low : crop_up, crop_low : crop_up]
    gt = cv2.resize(gt, (w, h))
    return gt