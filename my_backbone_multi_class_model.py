import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from mymrcnn.config import Config
import mymrcnn.myBackboneModel_Multi as modellib
import mymrcnn.ImageDataSet as dataSetlib
import mymrcnn.datagenerator as datagenerator
import pandas as pd
ROOT_DIR = os.path.abspath("D:/workfolder/myNewmrcnnClassWork_v2")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
def runTrainning(epochs):
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()        
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.xlsx",
                        WORK_DIR + '/data_train.xlsx',
                        WORK_DIR + '/data_val.xlsx')
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
            epochs=100, 
            layers='all')  
def runTesting():
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.xlsx",
                        WORK_DIR + '/data_train.xlsx',
                        WORK_DIR + '/data_val.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
    dataset_train.noMask = dataset_val.noMask = True
    model = modellib.MyBackboneModel(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(model.find_last(), by_name=True)
    image_ids = dataset_val._image_ids
    df_data = pd.read_excel(WORK_DIR + '/data_val.xlsx')
    df = pd.DataFrame(columns=["image_id","True_Gravel","True_Sugar","True_Fish","True_Flower","Pred_Gravel",\
        "Pred_Sugar","Pred_Fish","Pred_Flower","Match"],index=[])
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
        img_file_id = dataset_val.image_info[id]["id"]
        true_classes = df_data.loc[img_file_id].values[1:]
        classes = np.reshape(np.round(output[2]),[-1])
        Match = True
        for idx,v in enumerate(classes):
            if v != true_classes[idx]:
                Match = False
        if img_file_id not in df.index:
            df.loc[img_file_id] = [img_file_id] + true_classes.tolist() + classes.tolist() + [1. if Match else 0.]
        print(img_file_id)
    df.to_excel(WORK_DIR + "/MyMRCNN_WHOLE_Model_multi_class_pred_6970.xlsx")
# runTesting()
if __name__ == "__main__":  
     runTrainning(EPOCS)