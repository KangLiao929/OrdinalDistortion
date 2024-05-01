import tensorflow as tf
import keras.backend as K
import os
import argparse

from models.loss import *
from models.networks import OrdinalDistortionNet
from data.data_utils import *
from data.datasets import *
from utils.norm_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(opt):
    
    save_img_path = opt.save_img_path + opt.method + '/'
    os.makedirs(save_img_path, exist_ok=True)
    
    K.clear_session()
    model = OrdinalDistortionNet(opt.backbone_index)
    model.load_weights(opt.save_weights_path, by_name=True)
    
    src, pacth_t, block_t, file_name = get_test_img(opt.test_num, opt.test_path, opt.img_size, opt.img_size, 
                                                        opt.patch_size, opt.patch_size, opt.block_size, opt.block_size, 
                                                        patch_index=opt.patch_index)
    
    results = eva_correct(model, src, opt.test_num, pacth_t, block_t)
    for i in range(opt.test_num):
        cv2.imwrite(save_img_path + file_name[i], results[i])
    if(opt.test_label):
        '''if the GT distortion parameters are avaiable, further evaluate the MDLD metrics'''
        pacth_t, block_t, dp_t, dl_t = get_img(opt.test_num, opt.test_path, opt.patch_size, opt.patch_size, opt.block_size,
                                                opt.block_size)
        sum_mdld = eva_test(model, opt.test_num, pacth_t, block_t, dp_t, dl_t)
        print("MDLD metrics on test set: ", sum_mdld)
    
if __name__ == '__main__':

    print('<==================== setting arguments ===================>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_num", type=int, default=50, help="test data num")
    parser.add_argument("--test_label", type=bool, default=True, help="test from scratch (False) or labeled images (True)")
    parser.add_argument("--test_path", type=str, default="./dataset/test/A/*.jpg", help="path of the test dataset")
    parser.add_argument("--backbone_index", type=int, default=0, help="0: VGG16; 1: VGG19; 2: ResNet50; 3: InceptionV3; 4: Xception")
    parser.add_argument("--img_size", type=int, default=256, help="training image size")
    parser.add_argument("--patch_size", type=int, default=128, help="training patch size")
    parser.add_argument("--block_size", type=int, default=32, help="training block size")
    parser.add_argument("--patch_index", type=int, default=4, help="0, 1, 2, 3 for four arranged patches and 4 for random patches")
    parser.add_argument("--save_weights_path", default='./weights/OrdinalDistortionNet.h5', help="where to save models")
    parser.add_argument("--save_img_path", default='./results/', help="where to save results")
    parser.add_argument("--method", type=str, default="VGG16_ordinal", help="name of experiments")
    parser.add_argument("--gpu", type=str, default="0", help="gpu number")
    opt = parser.parse_args()
    print(opt)
    
    print('<==================== testing ===================>\n')
    
    test(opt)