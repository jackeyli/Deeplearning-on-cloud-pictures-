import cv2
import pandas as pd
import numpy as np
import sys
import json
import os
import shutil
from pathlib import Path
from collections import UserDict
from collections import deque
from sklearn.model_selection import train_test_split
from mrcnn import utils
basePath = "D:/MyWork/"
csv_data = pd.read_csv(basePath + "/train.csv",converters={'EncodedPixels': str})     

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
def randomSample(csv_data,train_path,val_path):
    train, val = train_test_split(csv_data, test_size=0.2)
    train.to_excel(train_path)
    val.to_excel(val_path)
def smallRandomSample(csv_data,train_path,val_path):
    train, small_set = train_test_split(csv_data, test_size=0.2)
    small_train,small_val = train_test_split(small_set,test_size=0.2)
    small_train.to_excel(train_path)
    small_val.to_excel(val_path)
# transformed = pd.read_excel(basePath + "/clean_train_images.xlsx")
# randomSample(transformed,basePath + "/clean_data_train.xlsx",basePath + "/clean_data_val.xlsx")
def transformData(csv_data,path):
    classes = ["Gravel","Sugar","Fish","Flower"]
    df = pd.DataFrame(columns=["image_id","Gravel","Sugar","Fish","Flower"],index=[])
    newRow = {"image_id":"","Gravel":0,"Sugar":0,"Fish":0,"Flower":0}
    for idx,row in csv_data.iterrows():
        img_key = row["Image_Label"].split(".")[0]
        class_name = row["Image_Label"].split("_")[1]
        encodedPixels = row["EncodedPixels"]
        if len(encodedPixels) > 0 and len(encodedPixels.split(" ")) > 1:
            if img_key not in df.index:
                df.loc[img_key] = [img_key,0,0,0,0]
            df.loc[img_key][class_name] = 1
        print(idx)
    df.to_csv(path)
# loadImageInfos()   
# transformData(csv_data,basePath + "transformed_train.csv")
def calculateMeanPixel(path):
    files = list(Path(path).glob("*.jpg"))
    #rgb
    meanPixels = [0.,0.,0.]
    for i in range(0,len(files)):
        print(str(files[i]))
        img = cv2.imread(str(files[i]))
        #bgr to rgb
        img[:,:,[0,2]] = img[:,:,[2,0]]
        meanPixels = np.add(np.mean(img,axis=(0,1)),meanPixels)
    meanPixels /= len(files)
    with Path(basePath + "meanPixels.json").open("w") as f:
        f.write(json.dumps(meanPixels))
# calculateMeanPixel(basePath + "train_images")
def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img
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
def createMask(isSep = False):
    for index, row in csv_data.iterrows():
        img_filename = row["Image_Label"].split("_")[0]
        label_name = row["Image_Label"].split("_")[1]
        file_ext = img_filename.split(".")[1]
        filename = img_filename.split(".")[0]
        msk_filename = filename + "_" + label_name + ".png"
        if not isinstance(row["EncodedPixels"],str):
            continue
        if len(row["EncodedPixels"]) == 0:
            continue
        encodedPixels = row["EncodedPixels"].split(" ")
        if len(encodedPixels) == 0:
            continue
        imgMask = rle_to_mask(row["EncodedPixels"],1400,2100)
        imgMask = cv2.resize(imgMask,(576,384),interpolation=cv2.INTER_NEAREST)
        if isSep:
            labeledMsk = createImageSingleMasks(imgMask)
            for i,mask in enumerate(labeledMsk):
                fName = filename + "_" + label_name + "_" + str(i) + ".png"
                print(fName)
                cv2.imwrite(basePath + "/masks_seperated/" + fName,mask)   
        else:
            fName = filename + "_" + label_name + "_" + str(i) + ".png"
            print(fName)
            cv2.imwrite(basePath + "/masks_shrinked/" + msk_filename,imgMask)    
# createMask() 
# createMask()
# def testCreateMask():
#     imgMask = cv2.imread(basePath + "/masks/1eefbee_Sugar.png")/
#     imgMask = cv2.resize(imgMask,(576,384),interpolation=cv2.INTER_NEAREST)
#     cv2.imwrite(basePath + "/testmasks/original.png",imgMask[:,:,0])  
#     labeledMsk = createImageSingleMasks(imgMask[:,:,0])
#     for i,mask in enumerate(labeledMsk):
#         fName = "result.png"
#         print(fName)
#         cv2.imwrite(basePath + "/testmasks/" + fName,mask)  
# testCreateMask()    
def deleteAbnormalMask():
    moveToPath = basePath + "/masktoremove"
    for item in Path(basePath + "/masks_seperated2").rglob('*.png'):
        filePath = str(item)
        img_mask_data = cv2.imread(filePath)
        img_mask_data = img_mask_data[:,:,[0]]
        bbox = utils.extract_bboxes(img_mask_data.astype(np.uint8))
        y1,x1,y2,x2 = bbox[0][0],bbox[0][1],bbox[0][2],bbox[0][3]
        width = x2 - x1
        height = y2 - y1
        print(filePath)
        if width * height < 2000:
            imgName = filePath.split(os.sep)[-1]
            shutil.move(filePath,moveToPath + "/" + imgName)
# deleteAbnormalMask()
def boxintersect(bbox1,bbox2):
    xcords1,xcords2,ycords1,ycords2 = [[bbox1[1],0],[bbox1[3],0]],[[bbox2[1],1],[bbox2[3],1]], \
        [[bbox1[0],0],[bbox1[2],0]],[[bbox2[0],1],[bbox2[2],1]]
    xcords = xcords1 + xcords2
    ycords = ycords1 + ycords2
    xcords.sort(key=lambda x:x[0])
    ycords.sort(key=lambda x:x[0])
    if xcords[0][1] ==  xcords[1][1] or ycords[0][1] == ycords[1][1]:
        return 0
    else:
       return (xcords[2][0] - xcords[1][0]) * (ycords[2][0] - ycords[1][0])
def boxU(bbox1,bbox2):
    xcords1,xcords2,ycords1,ycords2 = [[bbox1[1],0],[bbox1[3],0]],[[bbox2[1],1],[bbox2[3],1]], \
        [[bbox1[0],0],[bbox1[2],0]],[[bbox2[0],1],[bbox2[2],1]]
    xcords = xcords1 + xcords2
    ycords = ycords1 + ycords2
    xcords.sort(key=lambda x:x[0])
    ycords.sort(key=lambda x:x[0])
    return [ycords[0][0],xcords[0][0],ycords[-1][0],xcords[-1][0]]
def markAmbiguous(class_inst,class_name,imgs):
    cur_bbox = class_inst["bbox"]
    y1,x1,y2,x2 = cur_bbox[0],cur_bbox[1],cur_bbox[2],cur_bbox[3]
    cur_area = (x2 - x1) * (y2 - y1)
    for cls_name in imgs:
        if cls_name == class_name:
            continue
        cls_insts = imgs[cls_name]
        for idx,inst in enumerate(cls_insts):
            _bbox = inst["bbox"]
            _y1,_x1,_y2,_x2 = _bbox[0],_bbox[1],_bbox[2],_bbox[3]
            intersect = boxintersect(cur_bbox,_bbox)
            _area = (_x2 - _x1) * (_y2 - _y1)
            if intersect > _area * 0.8 or intersect > cur_area * 0.8:
                class_inst["ambi"] = True
                inst["ambi"] = True
    return False

def combineSameBBoxMask():
    maskdicts = {}
    for item in Path(basePath + "/masks_seperated").rglob('*.png'):
        filePath = str(item)
        img_id,img_class,class_instance = item.name.split("_")
        img_mask_data = cv2.imread(filePath)
        img_mask_data = img_mask_data[:,:,[0]]
        cur_bbox = utils.extract_bboxes(img_mask_data.astype(np.uint8))[0]
        y1,x1,y2,x2 = cur_bbox[0],cur_bbox[1],cur_bbox[2],cur_bbox[3]
        cur_area = (x2 - x1) * (y2 - y1)
        if img_id not in maskdicts.keys():
            img_inst = {}
            maskdicts[img_id] = img_inst
        if img_class not in img_inst.keys():
            img_cls = []
            img_inst[img_class] = img_cls
        if len(img_cls) == 0:
            new_Inst = {"path":[filePath],"bbox":cur_bbox,"ambi":False}
            img_cls.append(new_Inst)
        else:
            findInstance = False
            for instance in img_cls:
                inst_bbox = instance["bbox"]
                _y1,_x1,_y2,_x2 = inst_bbox[0],inst_bbox[1],inst_bbox[2],inst_bbox[3]
                if (abs(y1 - _y1) < 10 or abs(y2 - _y2) < 10) and (abs(x2 - _x1) < 60 or abs(x1 - _x2) < 60):
                    instance["bbox"] = boxU(cur_bbox,inst_bbox)
                    instance["path"].append(filePath)
                    findInstance = True
            if not findInstance:
                new_Inst = {"path":[filePath],"bbox":cur_bbox,"ambi":False}
                img_cls.append(new_Inst)
    for id in maskdicts:
        imgs = maskdicts[id]
        for cls_name in imgs:
            cls_insts = imgs[cls_name]
            for idx,inst in enumerate(cls_insts):
                img_data = np.zeros([384,576])
                for path in inst["path"]:
                    data = cv2.imread(path)
                    img_data = np.logical_or(img_data,data[:,:,0])
                img_data = img_data * 255
                _img_id,_img_class,_class_instance = inst["path"][0].split(os.sep)[-1].split("_")
                new_mask_file = _img_id + "_" + _img_class + "_" + str(idx)
                cv2.imwrite(basePath + "masks_seperated2/" + new_mask_file + ".png",img_data.astype(np.uint8))
                print(basePath + "masks_seperated2/" + new_mask_file + ".png")
# combineSameBBoxMask()

def createClipedImage():
    for item in Path(basePath + "/masks3").rglob('*.png'):
        filePath = str(item)
        img_id,img_class,class_instance = filePath.split(os.sep)[-1].split("_")
        img_file_name = filePath.split(os.sep)[-1]
        img_mask_data = cv2.imread(filePath)
        img_mask_data = img_mask_data[:,:,[0]]
        bbox = utils.extract_bboxes(img_mask_data.astype(np.uint8))[0]
        scaleX,scaleY = 2100 / 576,1400 / 384
        y1, x1, y2, x2 = int(bbox[0] * scaleY),int(bbox[1] * scaleX),int(bbox[2] * scaleY),int(bbox[3]* scaleX)
        width = x2 - x1
        height = y2 - y1
        if width / height < 0.4 or width / height > 2.5 :
            continue
        if width < 25 or height < 18:
            continue
        img = cv2.imread(basePath + "train_images/" + img_id + ".jpg")
        clippedImg = img[y1:y2, x1:x2,:]
        if width < height:
            clippedImg = np.transpose(clippedImg,axes=(1,0,2))
        clippedImg = cv2.resize(clippedImg,(576,384))
        cropped_img_file_name = basePath + "/train_class_images/" + img_file_name
        print(cropped_img_file_name)
        cv2.imwrite(cropped_img_file_name,clippedImg)
# createClipedImage()        
def createClipedImageSource():
    df_empty = pd.read_excel(basePath + "/clean_train_images.xlsx")
    classes = ["Gravel","Sugar","Fish","Flower"]
    newRow = ["",0,0,0,0]
    for item in Path(basePath + "/train_class_images").rglob('*.png'):
        filePath = str(item)
        img_id,img_class,class_instance = filePath.split(os.sep)[-1].split("_")
        img_key = filePath.split(os.sep)[-1].split(".")[0]
        if img_key not in df_empty.index:
            nr = newRow.copy()
            nr[0] = img_key
            nr[classes.index(img_class) + 1] = 1.
            df_empty.loc[img_key] = nr
    df_empty.to_excel(basePath + "/clean_train_images.xlsx")
def createSmallImage():
    for item in Path(basePath + "/train_images").rglob('*.jpg'):
        filePath = str(item)
        fileName = filePath.split(os.sep)[-1].split(".")[0] + ".jpg"
        img = cv2.imread(filePath)
        img = cv2.resize(img,(576,384))
        cv2.imwrite(basePath + "/train_image_shrinked" + os.sep + fileName,img)
        print(fileName)

def transformData(csv_data,path):
    classes = ["Gravel","Sugar","Fish","Flower"]
    df = pd.DataFrame(columns=["image_id","Gravel","Sugar","Fish","Flower"],index=[])
    newRow = {"image_id":"","Gravel":0,"Sugar":0,"Fish":0,"Flower":0}
    for idx,row in csv_data.iterrows():
        img_key = row["Image_Label"].split(".")[0]
        class_name = row["Image_Label"].split("_")[1]
        encodedPixels = row["EncodedPixels"]
        if len(encodedPixels) > 0 and len(encodedPixels.split(" ")) > 1:
            if img_key not in df.index:
                df.loc[img_key] = [img_key,0,0,0,0]
            df.loc[img_key][class_name] = 1
        print(idx)
    df.to_csv(path)

def createCombinedMaskForValTest():
    classes = ["Gravel","Sugar","Fish","Flower"]
    df = pd.read_excel(basePath + "/mrcnn_training_images.xlsx")
    for index,row in df.iterrows():
        img_key = row["image_id"]
        for clz in classes:
            if isinstance(row[clz],str):
                clzes = row[clz].split(" ")
                newMasks = np.full((384,576,1),0)
                for imgpath in clzes:
                    if len(imgpath) == 0:
                        continue
                    img = cv2.imread(basePath + "/masks_seperated2/" + imgpath)
                    newMasks = np.logical_or(newMasks,img[:,:,[0]])
                print(basePath + "seperate_masks_for_val_test/" + img_key + "_" + clz + ".png")
                cv2.imwrite(basePath + "seperate_masks_for_val_test/" + img_key + "_" + clz + ".png",newMasks.astype(np.uint8) * 255)
# createCombinedMaskForValTest()
def createNewTrainingXls():
    df = pd.DataFrame(columns=["image_id","Gravel","Sugar","Fish","Flower"],index=[])
    newRow = ["","","","",""]
    for item in Path(basePath + "/masks_seperated2").rglob('*.png'):
        filePath = str(item)
        print(filePath)
        img_id,img_class,class_instance = filePath.split(os.sep)[-1].split("_")
        if img_id not in df.index:
            df.loc[img_id] = newRow.copy()
            df.loc[img_id]["image_id"] = img_id
        df.loc[img_id][img_class] = df.loc[img_id][img_class] + " " + filePath.split(os.sep)[-1]
    df.to_excel(basePath + "/mrcnn_training_images.xlsx")
# createNewTrainingXls()
# mrcnn_transformed = pd.read_excel(basePath + "/mrcnn_training_images.xlsx")   
# randomSample(mrcnn_transformed,basePath + "/mrcnn_data_train.xlsx",basePath + "/mrcnn_data_val.xlsx")
# createSmallImage()
# createClipedImageSource()
def testCanny():
    image = cv2.imread("D:/MyWork/test_image.png")
    # cv2.imwrite("D:/MyWork/test_image_canny.png",cv2.Canny(image,100,200))
    canny = cv2.Canny(image,100,200)
    cv2.imwrite("D:/MyWork/test_image_canny.png",canny)
# testCanny()
def getAvgPixels():
    meanBlue,meanGreen,meanRed = 0,0,0
    img_count = 0
    for item in Path(basePath + "/train_images").rglob('*.jpg'):
        filePath = str(item)
        print(filePath)
        img = cv2.imread(filePath)
        meanBlue += np.sum(img[:,:,0]) / 2100 / 1400
        meanGreen += np.sum(img[:,:,1]) / 2100 / 1400
        meanRed += np.sum(img[:,:,2]) / 2100 / 1400
        img_count += 1
    meanBlue /= img_count
    meanGreen /= img_count
    meanRed /= img_count
    print(" blue " + str(meanBlue) + " green " + str(meanGreen) + " red " + str(meanRed))
getAvgPixels()
print("success!")

