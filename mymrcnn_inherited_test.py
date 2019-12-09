import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import myInheritedMrcnn.myConfig as cfg
import myInheritedMrcnn.mybackbonegraph as graph
import mrcnn.model as modellib
import mymrcnn.ImageDataSet as dataSetlib
import mrcnn.visualize as visualize
import matplotlib
import matplotlib.pyplot as plt
import mrcnn.utils as utils
import pandas as pd
ROOT_DIR = os.path.abspath("D:/workfolder/myInheritedmrcnnWork")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
INITIAL_MODEL_PATH = "D:/workfolder/initialWeightForMRCNN/mask_rcnn_mymrcnn_whole_model_0037.h5"
def compute_backbone_shapes(image_shape):
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in [4, 8, 16, 32, 32]]) 
class InferenceConfig(cfg.CloudPatternConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1   
def runTrainning(epochs):
    config = cfg.CloudPatternConfig()
    config.NAME = "MyMRCNN_Inherited_Model"
    config.BACKBONE = graph.dense_graph_simple_long
    config.COMPUTE_BACKBONE_SHAPE = compute_backbone_shapes
    config.IMAGE_SHAPE = [384,576,3]
    config.display()        
    dataSetFact = dataSetlib.ImageDataSetForMRCNNFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/mrcnn_training_images.xlsx",
                        WORK_DIR + '/mrcnn_data_train.xlsx',
                        WORK_DIR + '/mrcnn_data_val.xlsx')
    dataSetFact.preload_images()
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
# Validation dataset
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(INITIAL_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                 "mrcnn_bbox", "mrcnn_mask"])
    model.load_weights(model.find_last(), by_name=True)
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=150, 
            layers='heads')  
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
def runTesting():
    config = InferenceConfig()
    config.NAME = "MyMRCNN_Inherited_Model"
    config.BACKBONE = graph.dense_graph_simple_long
    config.COMPUTE_BACKBONE_SHAPE = compute_backbone_shapes
    config.IMAGE_SHAPE = [384,576,3]
    config.display() 
    df = pd.read_excel(WORK_DIR + "/mrcnn_data_val.xlsx")
    model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(INITIAL_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                 "mrcnn_bbox", "mrcnn_mask"])
    model.load_weights(model.find_last(), by_name=True)
    APs = []
    trueSum = 0
    predSum = 0
    intersects = 0
    for index,row in df.iterrows():
        img_key = row["image_id"]
        trueclzMasks = [np.full((384,576,1),0) for i in range(4)]
        predclzMasks = [np.full((384,576,1),0) for i in range(4)]
        for idx,clz in enumerate(["Gravel","Sugar","Fish","Flower"]):
            if isinstance(row[clz],str):
                clzes = row[clz].split(" ")
                print(WORK_DIR + "/seperate_masks_for_val_test/" + img_key + "_" + clz + ".png")
                clzMask = cv2.imread(WORK_DIR + "/seperate_masks_for_val_test/" + img_key + "_" + clz + ".png")
                clzMask = clzMask[:,:,[0]]
                clzMask = np.logical_and(clzMask,clzMask).astype(np.uint8)
                trueclzMasks[idx] = np.logical_or(trueclzMasks[idx],clzMask)
        image = cv2.imread(WORK_DIR + "/train_image_shrinked/" + img_key + ".jpg")
        results = model.detect([image],verbose = 0)
        r = results[0]
        masks,class_ids,scores = r['masks'],r['class_ids'],r['scores']
        for idx,clz_id in enumerate(class_ids):
            mask = masks[:,:,[idx]]
            predclzMasks[clz_id - 1] = np.logical_or(predclzMasks[clz_id - 1],mask)
        trueSum += np.sum(trueclzMasks)
        predSum += np.sum(predclzMasks)
        intersects += np.sum(np.logical_and(trueclzMasks,predclzMasks).astype(np.float32))
    score = 2 * intersects / (predSum + trueSum)
    print("The score is ",score)
runTesting()
# if __name__ == "__main__":  
#     runTrainning(EPOCS)