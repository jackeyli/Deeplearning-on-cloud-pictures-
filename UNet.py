import os
import json

import albumentations as albu
import cv2
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.exposure import adjust_gamma
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import tensorflow as tf

import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)  
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))
    
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks

def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)
        
        rle = mask2rle(mask)
        rles.append(rle)
        
    return rles
print("Yes")
train_df = pd.read_csv('D:/MyWork/train.csv')
train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

print(train_df.shape)
train_df.head()
mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
print(mask_count_df.shape)
mask_count_df.head()
sub_df = pd.read_csv('D:/MyWork/sample_submission.csv')
sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
MEAN_PIXEL = np.array([66.47, 71.05, 83.27,122.])
print("Done")
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path="D:/MyWork/train_images",
                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None, gamma=None,
                 augment=False, n_classes=4, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.reshape = reshape
        self.gamma = gamma
        self.n_channels = n_channels
        self.augment = augment
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.on_epoch_end()
        np.random.seed(self.random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            
            if self.augment:
                X, y = self.__augment_batch(X, y)
            
            return X, y
        
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        if self.reshape is None:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.reshape, self.n_channels))
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = self.base_path + "/" + im_name
            img = self.__load_rgb(img_path)
            
            if self.reshape is not None:
                img = np_resize(img, self.reshape)
            
            # Adjust gamma
            if self.gamma is not None:
                img = adjust_gamma(img, gamma=self.gamma)
            
            # Store samples
            X[i,] = img

        return X
    
    def __generate_y(self, list_IDs_batch):
        if self.reshape is None:
            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        else:
            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]
            
            rles = image_df['EncodedPixels'].values
            
            if self.reshape is not None:
                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)
            else:
                masks = build_masks(rles, input_shape=self.dim)
            
            y[i, ] = masks

        return y
    
    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img
    
    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # canny = cv2.Canny(img,100,200)
        # nImg = np.full((img.shape[0],img.shape[1],4),0.)
        # nImg[:,:,0:3] = img
        # nImg[:,:,3] = canny
        # nImg = nImg.astype(np.float32) / 255.
        return  img.astype(np.float32) / 255.
    
    def __random_transform(self, img, masks):
        composition = albu.Compose([
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1)
        ])
        
        composed = composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    
    def __augment_batch(self, img_batch, masks_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i, ], masks_batch[i, ] = self.__random_transform(
                img_batch[i, ], masks_batch[i, ])
        
        return img_batch, masks_batch
BATCH_SIZE = 24

train_idx, val_idx = train_test_split(
    mask_count_df.index, random_state=2019, test_size=0.1
)

train_generator = DataGenerator(
    train_idx, 
    df=mask_count_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    reshape=(320, 480),
    gamma=0.8,
    # gamma = None,
    augment=True,
    n_channels=3,
    n_classes=4
)

val_generator = DataGenerator(
    val_idx, 
    df=mask_count_df,
    target_df=train_df,
    batch_size=BATCH_SIZE, 
    reshape=(320, 480),
    gamma=0.8,
    # gamma = None,
    augment=False,
    n_channels=3,
    n_classes=4
)
model = sm.Unet(
    'resnet34', 
    classes=4,
    input_shape=(320, 480, 3),
    activation='sigmoid'
)
opt = keras.optimizers.SGD(
            lr=0.00025, momentum=0.9,
            clipnorm=5)
model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coef])
model.summary()
model.load_weights("model.h5")
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
es = EarlyStopping(monitor='val_dice_coef', min_delta=0.0001, patience=5, verbose=1, mode='max', restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.25, patience=2, verbose=1, mode='max', min_delta=0.0001)

# history = model.fit_generator(
#     train_generator,
#     validation_data=val_generator,
#     callbacks=[checkpoint, rlr,es],
#     epochs=1,
#     verbose=1,
# )

# history_df = pd.DataFrame(history.history)
# history_df.to_csv('history.csv', index=False)

# history_df[['loss', 'val_loss']].plot()
# history_df[['dice_coef', 'val_dice_coef']].plot()
# history_df[['lr']].plot()
def draw_convex_hull(mask, mode='convex'):
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        elif mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        elif mode=='approx':
            epsilon = 0.02*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.
def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num
from tta_wrapper import tta_segmentation

model = tta_segmentation(model, h_flip=True, v_flip=True,h_shift=(-10,10),v_shift=(-10,10),merge='mean')
best_threshold = 0.5
best_size = 15000
thresholds_size = [[0.25,7500],[0.3,12500],[0.3,7500],[0.25,5000]]
threshold = best_threshold
min_size = best_size

test_df = []
param_dict = {}
encoded_pixels = []
TEST_BATCH_SIZE = 500
# res_frame = []
# pooled_pred_masks = None
# for i in range(0,len(val_idx),TEST_BATCH_SIZE):
#     val_batch_idx = val_idx[i:min(len(val_idx),i + TEST_BATCH_SIZE)]
#     val_test_generator = ValDataGenerator(
#                 val_batch_idx, 
#                 shuffle=False,
#                 df=mask_count_df,
#                 target_df=train_df,
#                 mode='predict',
#                 batch_size=1, 
#                 reshape=(320, 480),
#                 gamma=0.8,
#                 # gamma = None,
#                 augment=False,
#                 n_channels=3,
#                 n_classes=4
#             )
#     batch_pred_masks = model.predict_generator(
#                 val_test_generator, 
#                 workers=1,
#                 verbose=1
#             )
#     if pooled_pred_masks is None:
#         pooled_pred_masks = batch_pred_masks
#     else:
#         pooled_pred_masks = np.concatenate((pooled_pred_masks,batch_pred_masks),axis=0)
# print("Done on pooled result")
# for s_thres in [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
#     for s_size in [2500,5000,7500,10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000]:    
#         thresholds_size[3] = [s_thres,s_size]
#         score = 0
#         count = 0
#         for i in range(0,len(val_idx),TEST_BATCH_SIZE):
#             val_test_generator
#             val_batch_idx = val_idx[i:min(len(val_idx),i + TEST_BATCH_SIZE)]
#             batch_pred_masks = pooled_pred_masks[i:i + TEST_BATCH_SIZE, ]
#             for j, idx in enumerate(val_batch_idx):
#                 filename = mask_count_df['ImageId'].iloc[idx]
#                 image_df = train_df[train_df['ImageId'] == filename].copy()
#                 # Batch prediction result set
#                 pred_masks = batch_pred_masks[j, ].copy()
#                 trueSum = 0
#                 predSum = 0
#                 intersects = 0
#                 for k in range(pred_masks.shape[-1]):
#                     if k != 3:
#                         continue
#                     pred_mask = pred_masks[...,k].astype('float32') 
#                     if pred_mask.shape != (350, 525):
#                         pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
#                     pred_mask, num_predict = post_process(pred_mask,thresholds_size[k][0], thresholds_size[k][1])
#                     if num_predict > 0:
#                         pred_mask = draw_convex_hull(pred_mask.astype(np.uint8)*255)
#                     rle = image_df['EncodedPixels'].values[k]
#                     reshaped_true = np.full((350,525),0.)
#                     if type(rle) is str:
#                         true_masks = rle2mask(rle,(1400,2100))
#                         reshaped_true = np_resize(true_masks, (350,525))
#                         trueSum = np.sum(reshaped_true.astype('float32'))
#                     predSum = np.sum(pred_mask.astype('float32'))
#                     intersects = np.sum(np.logical_and(reshaped_true.astype('bool'),pred_mask.astype('bool')).astype('float32'))
#                     if predSum + trueSum > 0:
#                         score += 2 * intersects / (predSum + trueSum)
#                         count += 1
#         param_dict_str = str(s_thres) + "_" + str(s_size)
#         print(param_dict_str,score / count)
#         res_frame += [[param_dict_str,score]]
# res_frame = pd.DataFrame(res_frame,columns=["Param","Score"])
# res_frame.head(100)
# res_frame.to_csv("val_params.csv",columns=["Param","Score"], index=False)
thresholds_size = [[0.45,7500],[0.5,12500],[0.45,7500],[0.4,5000]]
val_final_generator = DataGenerator(
    val_idx[0:200], 
    df=mask_count_df,
    target_df=train_df,
    shuffle=False,
    reshape=(320, 480),
    mode='predict',
    gamma=0.8,
    batch_size=1,
    # gamma = None,
    augment=False,
    n_channels=3,
    n_classes=4
)
batch_pred_masks = model.predict_generator(
        val_final_generator, 
        workers=1,
        verbose=1
)
classes = ["Fish","Flower","Gravel","Sugar"]
f_path = "D:/MyWork/pred_mask"
from sklearn.metrics import roc_auc_score
totalRate = 0
totalCounts = 0
totalROC = 0
total_true_masks = np.array([])
total_pred_masks = np.array([])
for j, idx in enumerate(val_idx[0:200]):
    filename = mask_count_df['ImageId'].iloc[idx]
    image_df = train_df[train_df['ImageId'] == filename].copy()
        
    # Batch prediction result set
    pred_masks = batch_pred_masks[j, ]
    temp_pred_masks = np.full((350,525,4),0)
    temp_true_masks = np.full((350,525,4),0)
    totalEquals = 0  
    for k in range(pred_masks.shape[-1]):
        pred_mask = pred_masks[...,k].astype('float32') 
            
        if pred_mask.shape != (350, 525):
            pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                
        pred_mask, num_predict = post_process(pred_mask, thresholds_size[k][0], thresholds_size[k][1])
        if num_predict > 0:
            pred_mask = draw_convex_hull(pred_mask.astype(np.uint8))
            temp_pred_masks[:,:,k] = pred_mask
        rle = image_df['EncodedPixels'].values[k]
        if type(rle) is str:
            true_masks = rle2mask(rle,(1400,2100))
            reshaped_true = np.round(np_resize(true_masks,(350,525)))
            temp_true_masks[:,:,k] = reshaped_true    
    total_true_masks = np.concatenate([total_true_masks,np.reshape(temp_true_masks,[-1])])
    total_pred_masks = np.concatenate([total_pred_masks,np.reshape(temp_pred_masks,[-1])])
    totalEquals = np.sum(np.equal(temp_pred_masks,temp_true_masks))
    totalPixels = 350 * 525 * 4
    totalRate += totalEquals / totalPixels
    totalCounts += 1
total_true_masks = np.reshape(np.array(total_true_masks),[-1])
total_pred_masks = np.reshape(np.array(total_pred_masks),[-1])
print("Acc avg",totalRate / totalCounts)
print("roc_auc avg",roc_auc_score(total_true_masks,total_pred_masks))
            #encoded_pixels.append(r)
# for i in range(0, test_imgs.shape[0], TEST_BATCH_SIZE):
#     batch_idx = list(
#         range(i, min(test_imgs.shape[0], i + TEST_BATCH_SIZE))
#     )

#     test_generator = DataGenerator(
#         batch_idx,
#         df=test_imgs,
#         shuffle=False,
#         mode='predict',
#         dim=(320, 480),
#         reshape=(320, 480),
#         n_channels=3,
#         gamma=0.8,
#         base_path='D:/MyWork/test_images',
#         target_df=sub_df,
#         batch_size=1,
#         n_classes=4
#     )

#     batch_pred_masks = model.predict_generator(
#         test_generator, 
#         workers=1,
#         verbose=1
#     ) 
#     # Predict out put shape is (320X480X4)
#     # 4  = 4 classes, Fish, Flower, Gravel Surger.
    
#     for j, idx in enumerate(batch_idx):
#         filename = test_imgs['ImageId'].iloc[idx]
#         image_df = sub_df[sub_df['ImageId'] == filename].copy()
        
#         # Batch prediction result set
#         pred_masks = batch_pred_masks[j, ]
        
#         for k in range(pred_masks.shape[-1]):
#             pred_mask = pred_masks[...,k].astype('float32') 
            
#             if pred_mask.shape != (350, 525):
#                 pred_mask = cv2.resize(pred_mask, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                
#             pred_mask, num_predict = post_process(pred_mask, thresholds_size[k][0], thresholds_size[k][1])
            
#             if num_predict == 0:
#                 encoded_pixels.append('')
#             else:
#                 pred_mask = draw_convex_hull(pred_mask.astype(np.uint8)*255)
#                 r = mask2rle(pred_mask)
#                 encoded_pixels.append(r)
                
# sub_df['EncodedPixels'] = encoded_pixels
# sub_df.to_csv('submission_presentation.csv', columns=['Image_Label', 'EncodedPixels'], index=False)