import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from mymrcnn.config import Config
import mymrcnn.myBackboneModel as modellib
import mymrcnn.ImageDataSet as dataSetlib
import mymrcnn.datagenerator as datagenerator
import pandas as pd
ROOT_DIR = os.path.abspath("D:/workfolder/myNewmrcnnClassWork_v2")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
WEIGHT_DIR = "D:/workfolder/myNewmrcnnClassWork_v2/mask_rcnn_mymrcnn_whole_model_0037.h5"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
def runTrainning(epochs):
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()        
    config.MEAN_PIXEL = np.array([66.47, 71.05, 83.27])
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/clean_train_images.xlsx",
                        WORK_DIR + '/clean_data_train.xlsx',
                        WORK_DIR + '/clean_data_val.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
    dataset_train.noMask = dataset_val.noMask = True
# Validation dataset
    model = modellib.MyBackboneModel(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(model.find_last(), by_name=True)
    learning_rate = 0.001
    model.train(dataset_train, dataset_val, 
            learning_rate=learning_rate, 
            epochs=50, 
            layers='all')  
def runTesting():
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/clean_train_images.xlsx",
                        WORK_DIR + '/clean_data_train.xlsx',
                        WORK_DIR + '/clean_data_val.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
    dataset_train.noMask = dataset_val.noMask = True
    model = modellib.MyBackboneModel(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(WEIGHT_DIR, by_name=True)
    image_ids = dataset_val._image_ids
    df = pd.DataFrame(columns=["image_id","Trueth","Pred","Match"],index=[])
    for id in image_ids:
        image, gt_class_ids, gt_masks = datagenerator.load_image_gt(dataset_val,config,id,augment=False,
                                augmentation=None,
                                use_mini_mask=config.USE_MINI_MASK) 
        batch_images = np.zeros(
                        (1,) + image.shape, dtype=np.float32)
        batch_gt_class_ids = np.zeros(
                    (1, config.NUM_CLASSES), dtype=np.int32)
        batch_gt_masks = np.zeros(
                    (1, gt_masks.shape[0], gt_masks.shape[1],
                     config.NUM_CLASSES), dtype=gt_masks.dtype)
        batch_images[0] = datagenerator.mold_image(image.astype(np.float32), config)
        output = model.keras_model.predict([batch_images,batch_gt_class_ids])
        class_id = np.argmax(output[0])
        class_names = ["Gravel","Sugar","Fish","Flower"]
        pred_class_name = class_names[class_id]
        img_file_id = dataset_val.image_info[id]["id"]
        img_true_class = img_file_id.split("_")[1]
        if img_file_id not in df.index:
            df.loc[img_file_id] = [img_file_id,img_true_class,pred_class_name,1. if img_true_class == pred_class_name else 0.]
        print(img_file_id)
    df.to_excel(WORK_DIR + "/MyMRCNN_WHOLE_Model_pred_MEAN_37_epoc.xlsx")
runTesting()
# if __name__ == "__main__":  
#      runTrainning(EPOCS)