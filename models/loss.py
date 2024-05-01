import keras.backend as K
import tensorflow as tf

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred-y_true), axis=-1)
    
def smoothL1(y_true, y_pred, HUBER_DELTA=0.5):
    x = K.abs(y_true - y_pred)
    x = K.switch(x<HUBER_DELTA, 0.5*x**2, HUBER_DELTA*(x-0.5*HUBER_DELTA))
    return K.sum(x)

def ordinary_loss(y_true, y_pred, num=4):
    weights = K.sum(K.relu(y_pred[:, :-1] - y_pred[:, 1:])) / num
    weights = K.sigmoid(weights) / 10.
    return (1.0+weights)*smoothL1(y_true, y_pred)