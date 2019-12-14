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
from mymrcnn import datagenerator
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

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)

    x = KL.Activation('relu')(x)
    
    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def resnet_graph(input_image,stage4=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    # Stage 1
    # x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(32, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    C2 = x = identity_block(x, 3, [16, 16, 64], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [32, 32, 128], stage=3, block='c', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [32, 32, 128], stage=4, block='a', train_bn=train_bn)
    block_count = 3
    for i in range(block_count):
        x = identity_block(x, 3, [32, 32, 128], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    return [C1, C2, C3, C4]

def generateBoxByScaleList(lst):
    boxes = []
    for scale in lst:
        for i in range(0,scale):
            for j in range(0,scale):
                boxes += [[i / scale, j / scale,(i + 1) / scale,(j + 1)/scale]]
    return boxes
class FeatureTransformLayer(KE.Layer):
    def __init__(self,**kwargs):
        super(FeatureTransformLayer, self).__init__(**kwargs)
    def call(self,inputs):
        feature_counts = len(inputs)
        mrcnn_features = tf.concat(inputs, axis=1)
        shapeP2 = K.int_shape(inputs[0])
        shape = [-1 if x is None else x for x in shapeP2[:1]] + [feature_counts] + \
        [-1 if x is None else x for x in shapeP2[1:]]
        output = tf.reshape(mrcnn_features,shape)
        return output
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + (len(input_shape),) + (input_shape[0][1:])

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
def concatFeatures(feature,featureSlice,axis):
    featureArr = tf.split(feature,featureSlice,axis=axis)
    squeezedArr = []
    for t in featureArr:
        squeezedArr.append(K.squeeze(t,axis=axis))
    output = tf.concat(squeezedArr,axis=-1)
    return output
class FlatConvLayer(KE.Layer):
    def __init__(self,**kwargs):
        super(FlatConvLayer, self).__init__(**kwargs)
    def call(self,inputs):
    #[batch 3 32 48 64 ]
        tensorArr = tf.split(inputs,[1,1,1],axis=1)
        squeezedArr = []
        for t in tensorArr:
            squeezedArr.append(K.squeeze(t,axis=1))
    # now add them together
    # [batch,32,48,64]
        output = KL.Add()(squeezedArr)
        cshape = K.int_shape(output)
        dim_2 = cshape[1] * cshape[2] * cshape[3]
        out = tf.reshape(output,[-1,dim_2])
        return out
    def compute_output_shape(self, input_shape):
        return (1,) + (input_shape[1] *input_shape[2] *input_shape[3],)
def mrcnn_mask_loss_graph(reshaped_input_gt_true_mask, pred_masks):
    return K.mean(K.binary_crossentropy(target=reshaped_input_gt_true_mask, output=pred_masks))
def msk_loss_graph(pred_mask,true_mask,num_classes,batchSize):
    reshaped_input_gt_mask = tf.reshape(true_mask,[-1])
    reshaped_pred_mask = tf.reshape(pred_mask,[-1])
    mask_loss = KL.Lambda(lambda x: msk_class_loss_graph(*x), name="mrcnn_mask_loss")(
                [reshaped_input_gt_mask, reshaped_pred_mask])
    return mask_loss
def msk_acc_graph(pred_mask,true_mask,num_classes,batchSize):
    reshaped_input_gt_mask = tf.reshape(true_mask,[-1])
    reshaped_pred_mask = tf.reshape(pred_mask,[-1])
    reshaped_input_gt_bool = tf.cast(reshaped_input_gt_mask,bool)
    reshaped_pred_mask_bool = tf.cast(K.round(reshaped_pred_mask),bool)
    logical_and = tf.logical_and(reshaped_input_gt_bool,reshaped_pred_mask_bool)
    logical_or = tf.logical_or(reshaped_input_gt_bool,reshaped_pred_mask_bool)
    sum_and = tf.reduce_sum(tf.cast(logical_and,tf.float32))
    sum_or = tf.reduce_sum(tf.cast(logical_or,tf.float32))
    return K.switch(sum_or > 0,sum_and / sum_or,tf.constant(1.0))
def dsc(y_true, y_pred):
     smooth = 0.5
     y_true_f = K.flatten(y_true)
     y_pred_f = K.flatten(y_pred)
     intersection = K.sum(y_true_f * y_pred_f)
     score = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
     return score
def dice_loss(y_true, y_pred):
    return (1 - dsc(y_true, y_pred))

def msk_class_loss_graph(y_true, y_pred):
    # return  K.mean(K.pow(y_pred,2.) * y_true * K.log(y_pred + 1e-9) \
    #         - K.pow((1.- y_pred),2.) * (1. - y_true) * K.log(1. - y_pred + 1e-9)) + \
    #         dice_loss(y_true,y_pred)
    return K.mean(K.binary_crossentropy(target=y_true, output=y_pred)) + \
        2 * dice_loss(y_true,y_pred)
class MyBackboneModel:
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.buildModel(mode,config)
    def buildModel(self,mode,config):
        input_image = KL.Input(
                shape=[384, 576, config.IMAGE_SHAPE[2]], name="input_image")
        #active classes in one_hot_coding
        input_gt_masks = KL.Input(
                    shape=[None,
                           None, config.NUM_CLASSES],
                    name="input_gt_masks")
        input_class_ids = KL.Input(
            shape=[config.NUM_CLASSES], name="input_image_class_id")

        _, C2, C3, C4 = resnet_graph(input_image,
                                             stage4=True, train_bn=config.TRAIN_BN)
        P2_Conv = KL.Conv2D(128, (3, 3),strides=(2,2),name='fpn_p2conv',padding="same")(C2)
        P2_Conv = BatchNorm(name='fpn_bn_p2conv')(P2_Conv, training=config.TRAIN_BN)
        P3_Cat = ConcatFeatureLayer(name="fpn_p2c3concat")([P2_Conv,C3])
        P3_Conv = KL.Conv2D(256, (3, 3),strides=(2,2),name='fpn_p3conv',padding="same")(P3_Cat)
        P3_Conv = BatchNorm(name='fpn_bn_p3conv')(P3_Conv, training=config.TRAIN_BN)
        P4_Cat = ConcatFeatureLayer(name="fpn_p3c4concat")([P3_Conv,C4])
        P4_Conv = KL.Conv2D(512, (3, 3), name='fpn_p4conv',padding="same")(P4_Cat)
        P4_Conv = BatchNorm(name='fpn_bn_p4conv')(P4_Conv, training=config.TRAIN_BN)

        P4 = KL.Lambda(lambda x:tf.image.crop_and_resize(x,np.full([config.BATCH_SIZE,4],[0.,0.,1.,1.]),np.arange(config.BATCH_SIZE),config.MASK_SHAPE))(P4_Conv)
        P3 = KL.Lambda(lambda x:tf.image.crop_and_resize(x,np.full([config.BATCH_SIZE,4],[0.,0.,1.,1.]),np.arange(config.BATCH_SIZE),config.MASK_SHAPE))(P3_Conv)
        P2 = KL.Lambda(lambda x:tf.image.crop_and_resize(x,np.full([config.BATCH_SIZE,4],[0.,0.,1.,1.]),np.arange(config.BATCH_SIZE),config.MASK_SHAPE))(P2_Conv)
        share = ConcatFeatureLayer(name="fpn_p2p3p4concat")([P2,P3,P4])
        mask_output = KL.Conv2D(config.NUM_CLASSES, (7, 7),padding="same",name="mrcnn_class_conv_mask_graph_7")(share)
        mask_output = KL.Activation('sigmoid')(mask_output)
        mask_loss = KL.Lambda(lambda x: msk_loss_graph(*x,config.NUM_CLASSES,config.BATCH_SIZE), name="mask_loss")(
                [mask_output,input_gt_masks])
        inputs = [input_image,input_class_ids,input_gt_masks]
        outputs = [mask_output,mask_loss]
        return KM.Model(inputs, outputs, name='mask_backbone')
        # mrcnn_features_raw = [P2,P3,P4]
        # mrcnn_features = FeatureTransformLayer(name="transform_pyramid_pooling_layer")(mrcnn_features_raw)
        # x = None
        # for i in range(2):
        #     x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="SAME"),
        #                    name="mrcnn_class_conv1_" + str(i))(mrcnn_features)
        #     x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1_' + str(i))(x, training=True)
        #     x = KL.Activation('relu')(x)
        # for i in range(4):
        #     x = KL.TimeDistributed(KL.Conv2D(64, (1, 1)),
        #                    name="mrcnn_class_conv2_" + str(i))(x)
        #     x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2_' + str(i))(x, training=True)
        #     x = KL.Activation('relu')(x)
        # shared = x
        # for i in range(8):
        #     mask_layer = KL.TimeDistributed(KL.Conv2D(32, (1, 1)),
        #                    name="mrcnn_class_conv3_" + str(i))(shared)
        #     mask_layer = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn3_' + str(i))(mask_layer, training=True)
        #     mask_layer = KL.Activation('relu')(mask_layer)
        

        # mask_layer = KL.TimeDistributed(KL.Conv2D(16, (1, 1)),
        #                    name="mrcnn_class_conv4")(shared)
        # mask_layer = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn4')(mask_layer, training=True)
        # mask_layer = KL.Activation('relu')(mask_layer)

        # mask_layer = KL.Lambda(lambda x:concatFeatures(x),name="concat_feacture_layer")(mask_layer)
    
    def compile(self,learning_rate, momentum):
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
                if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))
        self.keras_model.compile(
                optimizer=optimizer,
                loss=[None] * len(self.keras_model.outputs))
        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
        #add binary acc
            mask_input = self.keras_model.input[2]
            mask_pred = self.keras_model.output[0]
            self.keras_model.metrics_names.append("binary_acc_mask")
            bin_acc_mask = msk_acc_graph(mask_pred,mask_input,self.config.NUM_CLASSES,self.config.BATCH_SIZE)
            self.keras_model.metrics_tensors.append(bin_acc_mask)
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
    
    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)
    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = datagenerator.data_generator(train_dataset, self.config, shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE,
                                         no_augmentation_sources=no_augmentation_sources)
        val_generator = datagenerator.data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]
        
        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        #self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)


        
        
