# -*- coding: utf-8 -*-
from keras.layers import Input, Conv1D, Lambda, Dense, Flatten,MaxPooling1D, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
import os
import tensorflow as tf


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_siamese_net(input_shape = (2048,2)):
    left_input = Input(input_shape)
    right_input = Input(input_shape)  

    convnet = Sequential()
    convnet.add(tf.keras.layers.SeparableConv1D(14, 64, activation='relu',strides=8, depth_multiplier=29, padding='same',input_shape=input_shape))   # 16 , mul 29
    convnet.add(MaxPooling1D(strides=3))   
    convnet.add(tf.keras.layers.SeparableConv1D(32, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=3))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 2, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))               
    convnet.add(MaxPooling1D(strides=3  ))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(tf.keras.layers.SeparableConv1D(64, 3, activation='relu',strides=1, depth_multiplier=1, padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))
   						
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))

    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()

    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer) 

    return siamese_net

def load_wdcnn_net_depth(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    convnet = Sequential()

    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    #layer to merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    #call this layer on list of two input tensors.
    L1_distance = L1_layer([encoded_l, encoded_r])
    D1_layer = Dropout(0.5)(L1_distance)
    prediction = Dense(1,activation='sigmoid')(D1_layer)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = Adam()
    #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    
    return siamese_net


def load_wdcnn_net(input_shape = (2048,2),nclasses=10):
    left_input = Input(input_shape)
    convnet = Sequential()

    convnet.add(Conv1D(filters=16, kernel_size=64, strides=9, activation='relu', padding='same',input_shape=input_shape))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same'))                 
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same'))
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))   
    convnet.add(MaxPooling1D(strides=2))
    convnet.add(Flatten())
    convnet.add(Dense(100,activation='sigmoid'))

    encoded_cnn = convnet(left_input)
    prediction_cnn = Dense(10,activation='softmax')(Dropout(0.5)(encoded_cnn))
    wdcnn_net = Model(inputs=left_input,outputs=prediction_cnn)

    optimizer = Adam() 
    wdcnn_net.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    print(wdcnn_net.count_params())
    return wdcnn_net
