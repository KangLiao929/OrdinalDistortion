import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
import os
import numpy as np
import argparse

from models.loss import *
from models.networks import OrdinalDistortionNet
from data.data_utils import *
from data.datasets import *
from utils.norm_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(opt):
    
    weight_path = opt.save_weights_path + opt.method + '/'
    os.makedirs(weight_path, exist_ok=True)
    
    print("create models...")
    K.clear_session()
    model = OrdinalDistortionNet(opt.backbone_index)
    
    print("set optimizers...")
    k_opt = Adam(lr=opt.lr, beta_1=opt.b1, beta_2=opt.b2, epsilon=opt.ep)
    model.compile(optimizer=k_opt, loss=ordinary_loss, metrics=['mae'])
    
    print("Loading data...")
    pacth_t, block_t, dp_t, dl_t = get_img(opt.test_num, opt.test_path, opt.patch_size, opt.patch_size, opt.block_size, opt.block_size)

    print("Training...")
    iteration_num = opt.train_num / opt.batch_size
    for epoch in range(opt.epoch_num):
        print('epoch: {}/{}'.format(epoch, opt.epoch_num))
        loss = metrics = 0
        
        for _, (pacth_batch, block_batch, dp_batch, dl_batch) in enumerate(load_batch(opt.train_num, opt.batch_size, 
                                                                                            opt.train_path, opt.patch_size, opt.patch_size, 
                                                                                            opt.block_size, opt.block_size)):
            
            output = model.train_on_batch([pacth_batch, block_batch[:,0], block_batch[:,1], block_batch[:,2], block_batch[:,3]], dl_batch)
            loss += output[0]
            metrics += output[1]
        
        print("training loss and metrics: ", loss/iteration_num, metrics/iteration_num)
        K.set_value(model.optimizer.lr, get_lr(epoch, opt.lr))

        if ((epoch+1)%opt.save_interval==0):
            model.save_weights(weight_path + "OrdinalDistortionNet_{}.h5".format(epoch + 1))
            print("Evaluation...")
            sum_mdld = eva_test(model, opt.test_num, pacth_t, block_t, dp_t, dl_t)
            print("MDLD metrics on test set: ", sum_mdld)
    
if __name__ == '__main__':

    print('<==================== setting arguments ===================>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", type=int, default=300, help="epoch to start training from")
    parser.add_argument("--train_num", type=int, default=20000, help="training data num")
    parser.add_argument("--test_num", type=int, default=100, help="test data num")
    parser.add_argument("--train_path", type=str, default="./dataset/train/A/*.jpg", help="path of the train dataset")
    parser.add_argument("--test_path", type=str, default="./dataset/test/A/*.jpg", help="path of the test dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate of generator")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--ep", type=float, default=1e-08, help="adam: epsilon")
    parser.add_argument("--backbone_index", type=int, default=0, help="0: VGG16; 1: VGG19; 2: ResNet50; 3: InceptionV3; 4: Xception")
    parser.add_argument("--img_size", type=int, default=256, help="training image size")
    parser.add_argument("--patch_size", type=int, default=128, help="training patch size")
    parser.add_argument("--block_size", type=int, default=32, help="training block size")
    parser.add_argument("--save_interval", type=int, default=20, help="interval between saving image samples and checkpoints")
    parser.add_argument("--save_weights_path", default='./weights/', help="where to save models")
    parser.add_argument("--gpu", type=str, default="0", help="gpu number")
    parser.add_argument("--method", type=str, default="VGG16_ordinal", help="name of experiments")
    opt = parser.parse_args()
    print(opt)
    
    print('<==================== training ===================>\n')
    
    train(opt)

