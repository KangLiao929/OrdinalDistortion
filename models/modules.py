import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization, Input, add, concatenate, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D

def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')(input)
    x = BatchNormalization()(x)
    
    merged = add([input, x])
    return merged

def Res_net():
    patch = Input(shape=(32, 32, 256))
    patch_dwn = MaxPooling2D(pool_size=(2, 2), strides = None, padding='valid')(patch)
    r1 = res_block(patch_dwn, 256)
    r2 = res_block(r1, 256)
    r3 = res_block(r2, 256)
    r4 = res_block(r3, 256)
    r5 = res_block(r4, 256)
    
    model = Model(inputs = patch, outputs=r5)
    return model

'''Local siamese module (x4)'''
def LocalSiamese(res_net, input, kernel_size):
    fea = Conv2D(filters=64, kernel_size=kernel_size, padding='same')(input)
    fea = BatchNormalization()(fea)
    fea = Activation('relu')(fea)
    fea = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(fea)
    fea = BatchNormalization()(fea)
    fea = Activation('relu')(fea)
    fea = res_net(fea)
    fea = Flatten()(fea)
    fea = Dense(256)(fea)
    fea = BatchNormalization()(fea) 
    fea = Activation('relu')(fea)
    fea = Dropout(0.5)(fea)
    return fea

'''Global and local features fusion'''
def FeatureFusion(local_fea, global_fea):
    fea = concatenate([local_fea, global_fea], axis=-1)
    fea = Dense(256)(fea)
    fea = BatchNormalization()(fea) 
    fea = Activation('relu')(fea)
    fea = Dropout(0.5)(fea)
    return fea