
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("D:/workfolder")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import mymrcnn.ImageDataSet as dataSetlib
from mrcnn.model import log
import pandas as pd
ROOT_DIR = os.path.abspath("D:/workfolder/myInheritedmrcnnWork")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ProjectConfig(Config):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 576

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 500

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class InferenceConfig(ProjectConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1   
def runTrainning(epochs):
    config = ProjectConfig()
    config.NAME = "MyMRCNN_origin_Model"
    config.display()        
    dataSetFact = dataSetlib.ImageDataSetForMaskFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.xlsx",
                        WORK_DIR + '/data_train.xlsx',
                        WORK_DIR + '/data_val.xlsx')
    dataSetFact.preload_images()
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
# Validation dataset
    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50, 
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
    config.NAME = "MyMRCNN_origin_Model"
    config.display() 
    df = pd.read_excel(WORK_DIR + "/data_val.xlsx")
    model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])  
    model.load_weights(model.find_last(), by_name=True)
    trueSum = 0
    predSum = 0
    intersects = 0
    for index,row in df.iterrows():
        img_key = row["image_id"]
        trueclzMasks = [np.full((384,576,1),0) for i in range(4)]
        predclzMasks = [np.full((576,576,1),0) for i in range(4)]
        for idx,clz in enumerate(["Gravel","Sugar","Fish","Flower"]):
            print(WORK_DIR + "/masks_shrinked/" + img_key + "_" + clz + ".png")
            clzMask = cv2.imread(WORK_DIR + "/masks_shrinked/" + img_key + "_" + clz + ".png")
            if clzMask is None:
                continue
            clzMask = clzMask[:,:,[0]]
            trueclzMasks[idx] = clzMask
        image = cv2.imread(WORK_DIR + "/train_image_shrinked/" + img_key + ".jpg")
        image, window, scale, padding, crop = utils.resize_image(image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        results = model.detect([image],verbose = 0)
        r = results[0]
        masks,class_ids,scores = r['masks'],r['class_ids'],r['scores']
        for idx,clz_id in enumerate(class_ids):
            mask = masks[:,:,[idx]]
            predclzMasks[clz_id - 1] = np.logical_or(predclzMasks[clz_id - 1],mask)
        for i in range(4):
            predclzMasks[i] = predclzMasks[i][96:480,:,:]
        trueSum += np.sum(trueclzMasks)
        predSum += np.sum(predclzMasks)
        intersects += np.sum(np.logical_and(trueclzMasks,predclzMasks).astype(np.float32))
        score = 2 * intersects / (predSum + trueSum)
        print("The score is ",score)
# runTesting()
if __name__ == "__main__":  
    runTrainning(EPOCS)