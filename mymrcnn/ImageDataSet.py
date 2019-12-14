import cv2
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from collections import UserDict
from collections import deque
from mymrcnn import utils as utils
def generateClassVec(row):
    return np.array(row[1:].values)
class LimitedDict(UserDict):
    maxLen = sys.maxsize
    queue = deque([])
    def __init__(self,*args, **kwargs):
        super().__init__()
        if "maxLen" in kwargs:
            self.maxLen = max(kwargs["maxLen"],1)
    def __setitem__(self, key, value):
        if len(self) == self.maxLen:
            toDel = self.queue.pop()
            if super().__contains__(key):
                del self[toDel]
        self.queue.appendleft(key)
        if len(self.queue) > self.maxLen * 2:
            self.queue = deque(np.unique(self.queue))
        super().__setitem__(key, value)
    def __getitem__(self, key):
        if super().__contains__(key):
            self.queue.appendleft(key)
            if len(self.queue) > self.maxLen * 2:
                self.queue = deque(np.unique(self.queue))
        return super().__getitem__(key)


class ImageDataTrainningFactory():
    totalPath = ""
    trainPath = ""
    valPath = ""
    def __init__(self, workDir):
        self._image_ids = []
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.image_datas = LimitedDict(maxLen=200)
        self.image_mask_data = LimitedDict(maxLen=200)
        self.classes = ["Gravel","Sugar","Fish","Flower"]
        self.num_classes = len(self.classes)
        self.image_meta = {"width":2100,"height":1400,"MASKWIDTH":2100,"MASKHEIGHT":1400}
    def initialize(self,totalPath,trainPath,valPath):
        self.totalPath = totalPath
        self.trainPath = trainPath
        self.valPath = valPath
    def preload_images(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        if self.totalPath.split(".")[-1] == "csv":
            # csv_data = pd.read_csv(self.WORK_DIR + "/transformed_train.csv",converters={'image_id': str}) 
            csv_data = pd.read_csv(self.totalPath,converters={'image_id': str}) 
        else:
            csv_data = pd.read_excel(self.totalPath)
        for idx,row in csv_data.iterrows():
            self._image_ids += [row[0]]
            self.sub_image_info[row[0]] = generateClassVec(row)
    def getDataSet(self):
        # train_data = pd.read_excel(self.WORK_DIR + "/data_train_s.xlsx")
        train_data = pd.read_excel(self.trainPath)
        training_ids = []
        for idx,row in train_data.iterrows():
            training_ids += [row[0]]
        val_ids = []
        # val_data = pd.read_excel(self.WORK_DIR + "/data_val_s.xlsx")
        val_data = pd.read_excel(self.valPath)
        for idx,row in val_data.iterrows():
            val_ids += [row[0]]
        trainingSet = ImageDataSet(self.WORK_DIR)
        trainingSet.preload(training_ids,["Gravel","Sugar","Fish","Flower"])
        trainingSet.sub_image_info = self.sub_image_info.copy()
        valSet = ImageDataSet(self.WORK_DIR)
        valSet.preload(val_ids,["Gravel","Sugar","Fish","Flower"])
        valSet.sub_image_info = self.sub_image_info.copy()
        return trainingSet,valSet
class ImageDataSet(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, workDir):
        super().__init__()
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.IMAGE_DIR = self.WORK_DIR + "/train_image_shrinked"
        self.MASK_DIR = self.WORK_DIR + "/masks"
        self.image_datas = LimitedDict(maxLen=100)
        self.image_mask_data = LimitedDict(maxLen=100)
        self.image_meta = {"MASKHEIGHT":576,"MASKWIDTH":384}
        self.noMask = False
    def preload(self,image_ids,classes):
        self.classes = classes
        for i,v in enumerate(classes):
            self.add_class("clouds",i + 1,v)
        for i in image_ids:
            self.add_image("clouds", image_id=i, path=None,
                           width=2100, height=1400)
    def load_image(self, id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image_id = self.image_info[id]["id"]
        if image_id not in self.image_datas:
            if image_id.find("_") >= 0:
                image_file = self.IMAGE_DIR + "/" + image_id + ".png"
            else:
                image_file = self.IMAGE_DIR + "/" + image_id + ".jpg"
            #[width,height,3]
            image = cv2.imread(image_file)
            self.image_datas[image_id] = image
        return self.image_datas[image_id]
    def _load_mask(self,image_id,classIdx):
        classes = self.sub_image_info[image_id]
        image_meta = self.image_meta
        if classes[classIdx] and not self.noMask:
            img_mask_file_name = image_id + "_" + self.classes[classIdx] + ".png"
            img_mask = cv2.imread(self.MASK_DIR + "/" + img_mask_file_name)
            img_mask = np.logical_not(np.logical_not(img_mask[:,:,[0]])).astype(np.uint8)
            return img_mask
        else:
            return np.zeros([image_meta["MASKHEIGHT"],image_meta["MASKWIDTH"],1])
                
    def load_mask(self,id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_id = self.image_info[id]["id"]
        classes = self.sub_image_info[image_id]
        classIdxs = []
        for idx,v in enumerate(classes):
            classIdxs.append(idx)
        class_masks = [self._load_mask(image_id,i) for i in classIdxs]
        for i in range(1,len(class_masks)):
            class_masks[0] = np.concatenate((class_masks[0],class_masks[i]),axis=2)
        class_ids = np.array([self.class_names.index(self.classes[s]) for s in classIdxs])
        return class_masks[0].astype(np.uint8), class_ids.astype(np.int32)
    def load_class(self,id):
        image_id = self.image_info[id]["id"]
        return self.sub_image_info[image_id]

class ImageDataSetForMRCNNFactory():
    totalPath = ""
    trainPath = ""
    valPath = ""
    def __init__(self, workDir):
        self._image_ids = []
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.image_datas = LimitedDict(maxLen=200)
        self.image_mask_data = LimitedDict(maxLen=200)
        self.classes = ["Gravel","Sugar","Fish","Flower"]
        self.num_classes = len(self.classes)
        self.image_meta = {"width":576,"height":384,"MASKWIDTH":576,"MASKHEIGHT":384}
    def initialize(self,totalPath,trainPath,valPath):
        self.totalPath = totalPath
        self.trainPath = trainPath
        self.valPath = valPath
    def preload_images(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        csv_data = pd.read_excel(self.totalPath)
        for idx,row in csv_data.iterrows():
            self._image_ids += [row[0]]
            self.sub_image_info[row[0]] = {}
            for clzIdx,clz in enumerate(self.classes):
                self.sub_image_info[row[0]][self.classes[clzIdx]] = []
                if isinstance(row[clz],str) and len(row[clz].strip()) > 0:
                    ids = row[clz].strip().split(" ")
                    for msk in ids:
                        self.sub_image_info[row[0]][self.classes[clzIdx]].append(msk)
    def getDataSet(self):
        # train_data = pd.read_excel(self.WORK_DIR + "/data_train_s.xlsx")
        train_data = pd.read_excel(self.trainPath)
        training_ids = []
        for idx,row in train_data.iterrows():
            training_ids += [row[0]]
        val_ids = []
        # val_data = pd.read_excel(self.WORK_DIR + "/data_val_s.xlsx")
        val_data = pd.read_excel(self.valPath)
        for idx,row in val_data.iterrows():
            val_ids += [row[0]]
        trainingSet = ImageDataSetForMRCNN(self.WORK_DIR)
        trainingSet.preload(training_ids,["Gravel","Sugar","Fish","Flower"])
        trainingSet.sub_image_info = self.sub_image_info.copy()
        valSet = ImageDataSetForMRCNN(self.WORK_DIR)
        valSet.preload(val_ids,["Gravel","Sugar","Fish","Flower"])
        valSet.sub_image_info = self.sub_image_info.copy()
        return trainingSet,valSet


class ImageDataSetForMRCNN(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, workDir):
        super().__init__()
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.IMAGE_DIR = self.WORK_DIR + "/train_image_shrinked"
        self.MASK_DIR = self.WORK_DIR + "/masks_shrinked"
        self.image_datas = LimitedDict(maxLen=100)
        self.image_mask_data = LimitedDict(maxLen=100)
        self.image_meta = {"MASKHEIGHT":576,"MASKWIDTH":384}
        self.noMask = False
    def preload(self,image_ids,classes):
        self.classes = classes
        for i,v in enumerate(classes):
            self.add_class("clouds",i + 1,v)
        for i in image_ids:
            self.add_image("clouds", image_id=i, path=None,
                           width=576, height=384)
    def load_image(self, id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image_id = self.image_info[id]["id"]
        if image_id not in self.image_datas:
            image_file = self.IMAGE_DIR + "/" + image_id + ".jpg"
            image = cv2.imread(image_file)
            self.image_datas[image_id] = image
        return self.image_datas[image_id]
    def _load_mask(self,image_masks_file_name):
        img_mask = cv2.imread(self.MASK_DIR + "/" + image_masks_file_name)
        img_mask = np.logical_not(np.logical_not(img_mask[:,:,[0]])).astype(np.uint8)
        return img_mask
                
    def load_mask(self,id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_id = self.image_info[id]["id"]
        image_info = self.sub_image_info[image_id]
        lst_cls_masks = None
        class_ids = []
        for clz in self.classes:
            clzInfo = image_info[clz]
            if len(clzInfo) > 0:
                for path in clzInfo:
                    img_mask = self._load_mask(path)
                    if lst_cls_masks is None:
                        lst_cls_masks = img_mask
                    else:
                        lst_cls_masks = np.concatenate((lst_cls_masks,img_mask),axis=2)
                    class_ids.append(self.class_names.index(clz))
        return lst_cls_masks.astype(np.uint8), np.array(class_ids).astype(np.int32)

class ImageDataSetForMaskFactory():
    totalPath = ""
    trainPath = ""
    valPath = ""
    def __init__(self, workDir):
        self._image_ids = []
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.image_datas = LimitedDict(maxLen=200)
        self.image_mask_data = LimitedDict(maxLen=200)
        self.classes = ["Gravel","Sugar","Fish","Flower"]
        self.num_classes = len(self.classes)
        self.image_meta = {"width":576,"height":384,"MASKWIDTH":576,"MASKHEIGHT":384}
    def initialize(self,totalPath,trainPath,valPath):
        self.totalPath = totalPath
        self.trainPath = trainPath
        self.valPath = valPath
    def preload_images(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        if self.totalPath.split(".")[-1] == "csv":
            # csv_data = pd.read_csv(self.WORK_DIR + "/transformed_train.csv",converters={'image_id': str}) 
            csv_data = pd.read_csv(self.totalPath,converters={'image_id': str}) 
        else:
            csv_data = pd.read_excel(self.totalPath)
        for idx,row in csv_data.iterrows():
            self._image_ids += [row[0]]
            self.sub_image_info[row[0]] = generateClassVec(row)
    def getDataSet(self):
        # train_data = pd.read_excel(self.WORK_DIR + "/data_train_s.xlsx")
        train_data = pd.read_excel(self.trainPath)
        training_ids = []
        for idx,row in train_data.iterrows():
            training_ids += [row[0]]
        val_ids = []
        # val_data = pd.read_excel(self.WORK_DIR + "/data_val_s.xlsx")
        val_data = pd.read_excel(self.valPath)
        for idx,row in val_data.iterrows():
            val_ids += [row[0]]
        trainingSet = ImageDataSetForMask(self.WORK_DIR)
        trainingSet.preload(training_ids,["Gravel","Sugar","Fish","Flower"])
        trainingSet.sub_image_info = self.sub_image_info.copy()
        valSet = ImageDataSetForMask(self.WORK_DIR)
        valSet.preload(val_ids,["Gravel","Sugar","Fish","Flower"])
        valSet.sub_image_info = self.sub_image_info.copy()
        return trainingSet,valSet


class ImageDataSetForMask(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, workDir):
        super().__init__()
        self.sub_image_info = {}
        self.WORK_DIR = workDir
        self.IMAGE_DIR = self.WORK_DIR + "/train_image_shrinked"
        self.MASK_DIR = self.WORK_DIR + "/masks_shrinked"
        self.image_datas = LimitedDict(maxLen=100)
        self.image_mask_data = LimitedDict(maxLen=100)
        self.image_meta = {"MASKHEIGHT":576,"MASKWIDTH":384}
        self.noMask = False
    def preload(self,image_ids,classes):
        self.classes = classes
        for i,v in enumerate(classes):
            self.add_class("clouds",i + 1,v)
        for i in image_ids:
            self.add_image("clouds", image_id=i, path=None,
                           width=576, height=384)
    def load_image(self, id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image_id = self.image_info[id]["id"]
        img_data = np.full((384,576,4),0)
        image_file = self.IMAGE_DIR + "/" + image_id + ".jpg"
        image = cv2.imread(image_file)
        img_data[:,:,0:3] = image[:,:,:]
        img_data[:,:,3] = cv2.Canny(image,100,200)

        # image_id = self.image_info[id]["id"]
        # image_file = self.IMAGE_DIR + "/" + image_id + ".jpg"
        # image = cv2.imread(image_file)
        return img_data
    def _load_mask(self,image_masks_file_name):
        img_mask = cv2.imread(self.MASK_DIR + "/" + image_masks_file_name)
        img_mask = np.logical_not(np.logical_not(img_mask[:,:,[0]])).astype(np.uint8)
        return img_mask
                
    def load_mask(self,id):
        """Generate instance masks for shapes of the given image ID.
        """
        image_id = self.image_info[id]["id"]
        image_info = self.sub_image_info[image_id]
        masks = np.full((384,576,4),0)
        for idx,val in enumerate(image_info):
            if val == 1:
                clz = self.classes[idx]
                image_masks_file_name = image_id + "_" + clz + ".png"
                img_mask = self._load_mask(image_masks_file_name)
                masks[:,:,[idx]] = img_mask
        return masks.astype(np.float32), np.array([1.,1.,1.,1.]).astype(np.int32)
# def testDataSet():
#     DataSetFact = ImageDataTrainningFactory("D:/MyWork")
#     DataSetFact.preload_images()
#     trainingSet,valSet = DataSetFact.getDataSet()
#     trainingSet.prepare()
#     valSet.prepare()
#     img_my_img1 = trainingSet.load_image(0)
#     msk_my_mask1 = trainingSet.load_mask(0)
#     img_my_img2 = valSet.load_image(0)
#     msk_my_mask2 = valSet.load_mask(0)
#     mask_1 = msk_my_mask1[0][:,:,0] * 255
#     mask_2 = msk_my_mask1[0][:,:,1] * 255
#     mask_1_2 =  msk_my_mask2[0][:,:,0] * 255
#     mask_2_2 = msk_my_mask2[0][:,:,1] * 255
#     print(len(trainingSet._image_ids))
#     print(len(valSet._image_ids))
#     cv2.imwrite("D:/MyWork/test_read1.png",img_my_img1)
#     cv2.imwrite("D:/MyWork/test_read2.png",img_my_img2)
#     cv2.imwrite("D:/MyWork/msk_G1.png",mask_1)
#     cv2.imwrite("D:/MyWork/msg_S1.png",mask_2)
#     cv2.imwrite("D:/MyWork/msk_G2.png",mask_1_2)
#     cv2.imwrite("D:/MyWork/msg_S2.png",mask_2_2)
# testDataSet()