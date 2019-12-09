import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from mymrcnn.config import Config
import mymrcnn.myMrcnnModel as modellib
import mymrcnn.ImageDataSet as dataSetlib
import mymrcnn.datagenerator as datagenerator
ROOT_DIR = os.path.abspath("D:/workfolder/myNewmrcnnWork")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
def runTrainning(epochs):
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()        
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.csv",
                        WORK_DIR + '/data_train_s.xlsx',
                        WORK_DIR + '/data_val_s.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
# Validation dataset
    model = modellib.MyBackboneModel(mode="training", config=config,
                          model_dir=MODEL_DIR)
    # model.load_weights(model.find_last(), by_name=True)
    learning_rate = 0.001
    model.train(dataset_train, dataset_val, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            layers='all')  
def runTesting():
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()
    dataSetFact = dataSetlib.ImageDataTrainningFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.csv",
                        WORK_DIR + '/data_train_s.xlsx',
                        WORK_DIR + '/data_val_s.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
    image, gt_class_ids, gt_masks = datagenerator.load_image_gt(dataset_val,config,1,augment=False,
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
    model = modellib.MyBackboneModel(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(model.find_last(), by_name=True)
    output = model.keras_model.predict([batch_images,batch_gt_class_ids,batch_gt_masks])

    masks = output[0][0]
    masks = np.round(masks) * 255
    masks = cv2.resize(masks,(2100,1400))
    masklst = []
    class_names = ["Gravel","Sugar","Fish","Flower"]
    for i in range(masks.shape[-1]):
        masklst.append(masks[:,:,i])
    
    for i,name in enumerate(class_names):
        fname = dataset_val.image_info[1]["id"] + "_" + name + ".png"
        cv2.imwrite(WORK_DIR + "/" + fname,masklst[i])
    print(output[0].shape)
    print('hai')
runTesting()
# if __name__ == "__main__":  
#      runTrainning(EPOCS)