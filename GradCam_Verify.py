from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
from keras.models import Model
from mymrcnn.config import Config
import mymrcnn.myBackboneModel as modellib
ROOT_DIR = os.path.abspath("D:/workfolder/mymrcnnWork")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
def target_category_loss(x, tensor, nb_classes):
    return tf.multiply(x, tf.convert_to_tensor(tensor,dtype=tf.float32))

def load_image(path):
    img = cv2.imread(path)
    x = cv2.resize(img,(576,384))
    return x
 
def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)
 
def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    keras.layers.add
    return K.function([input_img, K.learning_phase()], [saliency])
 
def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):
 
        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]
 
        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu
 
        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model
 
def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
 
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
 
    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def _compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var)
          for var, grad in zip(var_list, grads)]
def target_category_loss_output_shape(input_shape):
    return input_shape
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def concatenateMatrix(tList,rows,cols):
    rowOutput = []
    for i in range(0,rows):
        tmp = []
        for j in range(0,cols):
            tmp.append(tList[i * rows + j])
        rowOutput.append(np.concatenate(tmp,axis=1))
    finalOutput = np.concatenate(rowOutput,axis = 0)
    return finalOutput
def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 4
    target_layer = lambda x: target_category_loss(x,category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output[0])
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input[0]], [conv_output, grads])
    
    output, grads_val = gradient_function([tf.convert_to_tensor(np.array([image]),dtype=tf.float32)])
    output, grads_val = output[0,0,:,:,:], grads_val[0,0,:, :, :]
    
    #85 * 7 7 32
    #64 16 4 1
    output_1 = output[0:64 * 7,:,:]
    output_2 = output[64 * 7:80 * 7,:,:]
    output_3 = output[80 * 7:84 * 7,:,:]
    output_4 = output[84*7:,:,:]
    output_1 = np.array_split(output_1,64,axis=0)
    output_1 = concatenateMatrix(output_1,8,8)
    output_2 = np.array_split(output_2,16,axis=0)
    output_2 = concatenateMatrix(output_2,4,4)
    output_3 = np.array_split(output_3,4,axis=0)
    output_3 = concatenateMatrix(output_3,2,2)
    grad_val_1 = grads_val[0:64 * 7,:,:]
    grad_val_2 = grads_val[64 * 7:80 * 7,:,:]
    grad_val_3 = grads_val[80 * 7:84 * 7,:,:]
    grad_val_4 = grads_val[84*7:,:,:]
    grad_val_1 = np.array_split(grad_val_1,64,axis=0)
    grad_val_1 = concatenateMatrix(grad_val_1,8,8)
    grad_val_2 = np.array_split(grad_val_2,16,axis=0)
    grad_val_2 = concatenateMatrix(grad_val_2,4,4)
    grad_val_3 = np.array_split(grad_val_3,4,axis=0)
    grad_val_3 = concatenateMatrix(grad_val_3,2,2)
    
    grad_val_1 = np.mean(cv2.resize(grad_val_1,(576,384)),axis=(0,1))
    grad_val_2 = np.mean(cv2.resize(grad_val_2,(576,384)),axis=(0,1))
    grad_val_3 = np.mean(cv2.resize(grad_val_3,(576,384)),axis=(0,1))
    grad_val_4 = np.mean(cv2.resize(grad_val_4,(576,384)),axis=(0,1))

    #first lets try to add grad_val_1 which is most precisely
    output_2 = cv2.resize(output_2,output_1.shape[0 : 2])
    output_3 = cv2.resize(output_3,output_1.shape[0 : 2])
    output_4 = cv2.resize(output_4,output_1.shape[0 : 2])
    weights = [grad_val_1,grad_val_2,grad_val_3,grad_val_4]
    outputs = [output_1,output_2,output_3,output_4]
    cam = np.ones(output_1.shape[0 : 2], dtype = np.float32)
    
    for j,weight in enumerate(weights):
        for i,w in enumerate(weight):
            cam += w * outputs[j][:, :, i]
    cam = cv2.resize(cam, (576, 384))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
 
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
"""
def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap """
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
preprocessed_input = load_image("D:/MyWork/train_images/0cabeda.jpg")

config = Config()
config.NAME = "MyMRCNN_Model"
config.display()        
# Validation dataset
model = modellib.MyBackboneModel(mode="training", config=config,
                        model_dir=MODEL_DIR)
model.load_weights(model.find_last(), by_name=True)
k_model = model.keras_model
predicted_class = 1
cam, heatmap = grad_cam(k_model, preprocessed_input, [0.,1.,0.,0.], "mrcnn_class_bn2_shared")

cv2.imwrite("D:/MyWork/gradcam_cloud.jpg", cam)
