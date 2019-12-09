import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from mymrcnn.config import Config
import mymrcnn.MyMaskModel_origin as modellib
import mymrcnn.ImageDataSet as dataSetlib
import mymrcnn.datagenerator as datagenerator
import pandas as pd
import mymrcnn.utils as utils
ROOT_DIR = os.path.abspath("D:/workfolder/myMaskmrcnnWork")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WORK_DIR = "D:/MyWork"
EPOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
def generateClassVec(row):
    return np.array(row[1:].values)
def runTrainning(epochs):
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()        
    dataSetFact = dataSetlib.ImageDataSetForMaskFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.xlsx",
                        WORK_DIR + '/data_train.xlsx',
                        WORK_DIR + '/data_val.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
# Validation dataset
    model = modellib.MyBackboneModel(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.keras_model.summary()
    model.load_weights(model.find_last())
    learning_rate = 0.001
    model.train(dataset_train, dataset_val, 
            learning_rate=learning_rate, 
            epochs=100, 
            layers='all')  

class Interval:
    def __init__(self,start,end):
        self.start = start
        self.end = end
class DSU:
    def __init__(self):
        self.parentMap = {}
        self.max_id = 0
    def putParent(self,interval,parent):
        if interval == parent:
            interval.id = self.max_id + 1
            self.max_id += 1
        self.parentMap[interval] = parent
    def getParent(self,interval):
        parent = self.parentMap[interval]
        if parent != interval:
            return self.getParent(parent)
        else:
            return parent
    def unionInterval(self,a,b):
        parentA = self.getParent(a)
        parentB = self.getParent(b)
        if parentA == parentB:
            return
        if parentA.id < parentB.id:
            self.putParent(parentB,parentA)
        else:
            self.putParent(parentA,parentB)

def createImageSingleMasks(imgs):
    curlabel = 0
    line_intervals = []
    dsu = DSU()
    for i in range(len(imgs)):
        intervals_in_line = []
        curInterval = [-1,-1]
        for j in range(len(imgs[i])):
            if imgs[i][j] != 0:
                if curInterval[0] == -1:
                    curInterval[0] = j
            if (imgs[i][j] == 0 or j == (len(imgs[i]) - 1)) and curInterval[0] != -1:
                if imgs[i][j] == 0:
                    curInterval[1] = j - 1
                else:
                    curInterval[1] = j
                interval = Interval(curInterval[0],curInterval[1])
                dsu.putParent(interval,interval)
                intervals_in_line.append(interval)
                findConnection = False
                if i > 0:
                    for intv in line_intervals[i - 1]:
                        if not(intv.start > interval.end or intv.end < interval.start):
                            findConnection = True
                            dsu.unionInterval(intv,interval)
                curInterval = [-1,-1]
        line_intervals.append(intervals_in_line)
    labelImgs = {}
    for row,line_interval in enumerate(line_intervals):
            for interval in line_interval:
                label_id = dsu.getParent(interval)
                if label_id not in labelImgs:
                    labelImgs[label_id] = np.full((len(imgs),len(imgs[0])),0)
                labelImgs[label_id][row,interval.start:interval.end + 1] = 1
    res = []
    for key in labelImgs:
        if np.sum(labelImgs[key]) > 400:
            res.append(labelImgs[key] * 255)
    return res
def postProcess(img,origin_img):
    labeledMsk = createImageSingleMasks(img)
    finalMsk = np.full((384,576,1),0.)
    for i,mask in enumerate(labeledMsk):
        nMask = mask.astype(np.uint8)[:,:,np.newaxis]
        bbox = utils.extract_bboxes(nMask)
        y1,x1,y2,x2 = bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3]
        finalMsk[y1:y2,x1:x2,[0]] = 1.
    for i in range(len(origin_img)):
        for j in range(len(origin_img[0])):
            if origin_img[i][j][0] <= 20 and origin_img[i][j][1] <= 20 and origin_img[i][j][2] <= 20:
                finalMsk[i][j][0] = 0.
    return finalMsk
def runTesting():
    config = Config()
    config.NAME = "MyMRCNN_WHOLE_Model"
    config.display()        
    dataSetFact = dataSetlib.ImageDataSetForMaskFactory(WORK_DIR)
    dataSetFact.initialize(WORK_DIR + "/transformed_train.xlsx",
                        WORK_DIR + '/data_train.xlsx',
                        WORK_DIR + '/data_val.xlsx')
    dataSetFact.preload_images()
    dataset_train,dataset_val = dataSetFact.getDataSet()
    dataset_train.prepare()
    dataset_val.prepare()
# Validation dataset
    df = pd.read_excel(WORK_DIR + "/data_val.xlsx")
    model = modellib.MyBackboneModel(mode="training", config=config,
                          model_dir=MODEL_DIR)
    model.load_weights(model.find_last(), by_name=True)
    totalScore = 0
    counts = 0
    classes = ["Gravel","Sugar","Fish","Flower"]
    trueSum = 0
    predSum = 0
    intersects = 0
    for index,row in df.iterrows():
        img_key = row["image_id"]
        trueclzMasks = np.full((384,576,4),0.)
        inputmasks = np.full((24,36,4),0)
        classVect = generateClassVec(row)
        for idx,clz in enumerate(classes):
            if classVect[idx] == 1:
                print(WORK_DIR + "/masks_shrinked/" + img_key + "_" + clz + ".png")
                clzMask = cv2.imread(WORK_DIR + "/masks_shrinked/" + img_key + "_" + clz + ".png")
                clzMask = clzMask[:,:,[0]]
                clzMask = np.logical_and(clzMask,clzMask).astype(np.float32)
                trueclzMasks[:,:,[idx]] = clzMask
        image = cv2.imread(WORK_DIR + "/train_image_shrinked/" + img_key + ".jpg")
        image = cv2.resize(image,(config.IMAGE_MAX_DIM,config.IMAGE_MIN_DIM),interpolation=cv2.INTER_LINEAR)
        outputs = model.keras_model.predict([[image],[inputmasks]])
        logits = outputs[0]
        predclzMasks = logits[0]
        resizedPredClzMasks = cv2.resize(predclzMasks.astype(np.float32),(576,384),interpolation=cv2.INTER_NEAREST)
        resizedPredClzMasks[resizedPredClzMasks >= 0.3] = 1.
        resizedPredClzMasks[resizedPredClzMasks < 0.3] = 0.
        # for i in range(4):
        #     resizedPredClzMasks[:,:,[i]] = postProcess(resizedPredClzMasks[:,:,[i]],image)
        trueSum += np.sum(trueclzMasks)
        predSum += np.sum(resizedPredClzMasks)
        intersects += np.sum(np.logical_and(resizedPredClzMasks,trueclzMasks).astype(np.float32))
        print("The score is ",2 * intersects / (predSum + trueSum))
# runTesting()
if __name__ == "__main__":  
     runTrainning(EPOCS)