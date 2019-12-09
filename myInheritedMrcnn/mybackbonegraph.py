import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

class ConcatFeatureLayer(KE.Layer):
    def __init__(self,**kwargs):
        super(ConcatFeatureLayer, self).__init__(**kwargs)
    def call(self,inputs):
        feature_counts = len(inputs)
        mrcnn_features = tf.concat(inputs, axis=-1)
        return tf.concat(inputs, axis=-1)
    def compute_output_shape(self, input_shape):
        sum_Of_dim = 0
        for i in range(len(input_shape)):
            sum_Of_dim += input_shape[i][-1]
        return (input_shape[0][0],) + input_shape[0][1:3] + (sum_Of_dim,)

def connectedConv(input_tensor,kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True,net_name="dense_res"):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = net_name + str(stage) + block + '_branch'
    bn_name_base = net_name + "_bn" + str(stage) + block + '_branch'
    L1 = input_tensor
    x = KL.Conv2D(nb_filter1, (1, 1),name=conv_name_base + '2a',
                  use_bias=use_bias)(L1)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    dense = ConcatFeatureLayer(name=conv_name_base + '_l1l2l3xconcat')([L1,x])
    kenerls = K.int_shape(L1)[-1] + K.int_shape(x)[-1]
    dense = KL.Conv2D(kenerls,(kernel_size,kernel_size),
            padding='same',name=conv_name_base + "_final",strides=strides,use_bias=use_bias)(dense)
    dense = BatchNorm(name=bn_name_base + '_final')(dense, training=train_bn)
    res = KL.Activation('relu', name=net_name + str(stage) + block + '_out')(dense)
    return res
def connectedIdentity(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True,net_name="dense_res"):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = net_name + str(stage) + block + '_branch'
    bn_name_base = net_name + '_bn' + str(stage) + block + '_branch'
    L1 = input_tensor
    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(L1)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    dense = ConcatFeatureLayer(name=conv_name_base + '_l1l2l3xconcat')([L1,x])
    res = KL.Activation('relu', name=net_name + str(stage) + block + '_out')(dense)
    return res
def dense_graph_simple_long(input_image,stage5=True,train_bn=True,net_name="dense_res_l"):
    # Stage 1
    x = KL.Conv2D(8, (7, 7), strides=(2, 2), name=net_name + 'conv1', use_bias=True,padding="same")(input_image)
    x = BatchNorm(name= net_name + 'bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((7, 7), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = connectedConv(x, 7, [8, 8, 32], stage=2, block='a', strides=(1, 1), train_bn=train_bn,net_name=net_name)
    x = KL.Dropout(rate=0.2,name=net_name + "dense_dropout2")(x)
    C2 = x = connectedIdentity(x, 7, [8, 8, 32], stage=2, block='b', strides=(1, 1), train_bn=train_bn,net_name=net_name)
    # Stage 3
    x = connectedConv(x, 7, [16, 16, 64], stage=3, block='a', train_bn=train_bn,net_name=net_name)
    x = KL.Dropout(rate=0.2,name=net_name +"dense_dropout3")(x)
    C3 = x = connectedIdentity(x, 7, [16, 16, 64], stage=3, block='b', strides=(1, 1), train_bn=train_bn,net_name=net_name)
    # Stage 4
    x = connectedConv(x, 7, [32, 32, 128], stage=4, block='a', train_bn=train_bn,net_name=net_name)
    x = KL.Dropout(rate=0.2,name= net_name + "dense_dropout4")(x)
    C4 = x = connectedIdentity(x, 7, [32, 32, 128], stage=4, block='b', train_bn=train_bn,net_name=net_name)
    # Stage 5
    if stage5:
        x = connectedIdentity(x, 7, [64, 64, 256], stage=5, block='a', train_bn=train_bn,net_name=net_name)
        x = KL.Dropout(rate=0.2,name= net_name + "dense_dropout5")(x)
        C5 = x = connectedIdentity(x, 7, [64, 64, 256], stage=5, block='b', train_bn=train_bn,net_name=net_name)
    else:
        C5 = None
    return [C1, C2, C3,C4,C5]

