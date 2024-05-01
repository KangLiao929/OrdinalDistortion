import keras
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Input, add, concatenate, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from models.modules import Res_net, LocalSiamese, FeatureFusion

def OrdinalDistortionNet(backbone_index, patch_shape=(128,128,3), block_shape=(32,32,4)):
    input_img = Input(shape=patch_shape)     #128x128
    patch1 = Input(shape=block_shape)        #32x32
    patch2 = Input(shape=block_shape)        #32x32
    patch3 = Input(shape=block_shape)        #32x32
    patch4 = Input(shape=block_shape)        #32x32
    
    if backbone_index == 0:
        base_model = VGG16(weights='imagenet', include_top=False)
    elif backbone_index == 1:
        base_model = VGG19(weights='imagenet', include_top=False)
    elif backbone_index == 2:
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif backbone_index == 3:
        base_model = InceptionV3(weights='imagenet', include_top=False)
    else:
        base_model = Xception(weights='imagenet', include_top=False)
    
    '''Global perception module'''     
    g = base_model(input_img)
    g = GlobalAveragePooling2D()(g)
    g = Dense(1024)(g)
    g = BatchNormalization()(g) 
    g = Activation('relu')(g)
    g = Dropout(0.5)(g)
    
    '''Local siamese module (x4)'''
    # shared weights pyramid residual module
    res_net = Res_net()
    # local features (output: 8x8x256)
    l1 = LocalSiamese(res_net, patch1, kernel_size=(1, 1))
    l2 = LocalSiamese(res_net, patch2, kernel_size=(3, 3))
    l3 = LocalSiamese(res_net, patch3, kernel_size=(5, 5))
    l4 = LocalSiamese(res_net, patch4, kernel_size=(7, 7))

    '''Global and local features fusion'''
    gl1 = FeatureFusion(l1, g)
    gl2 = FeatureFusion(l2, g)
    gl3 = FeatureFusion(l3, g)
    gl4 = FeatureFusion(l4, g)
    
    '''Distortion estimation module'''
    # final fusion
    gl = concatenate([gl1, gl2, gl3, gl4], axis=-1)
    # estimation
    x1 = Dense(512)(gl)
    x1 = BatchNormalization()(x1) 
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x2 = Dense(256)(x1)
    x2 = BatchNormalization()(x2) 
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x3 = Dense(4, activation='linear')(x2)
    
    model = Model(inputs = [input_img, patch1, patch2, patch3, patch4], outputs=x3)
    model.summary()
    return model