import os
import io
import pickle
import ImageOperations as imgOp
import numpy as np
import Params
import scipy.ndimage as sciImg
from sklearn import model_selection

class DataAccess(object):
    FileExtension = ".png"

    def __init__(self, imagesDir=Params.IMAGES_DIR, masksDir=Params.MASKS_DIR):
        self.images_gray = {}
        self.masks = {}
        self.Ids = []
        self.imagesDir = imagesDir
        self.masksDir = masksDir

        for imageFile in os.listdir(self.imagesDir):
            id = DataAccess.GetImageId(imageFile)
            self.Ids.append(id)
            self.images_gray[id] = (imgOp.ToGrayImage(imgOp.ImageToArrayConverter(os.path.join(self.imagesDir,imageFile))))
            maskPath = os.path.join(self.masksDir,imageFile)
            self.masks[id] = imgOp.ImageToArrayConverter(maskPath)
            self.masks[id] = self.masks[id].reshape((self.masks[id].shape[0],self.masks[id].shape[1],1))

        return super().__init__()


    def GetDataGenerators(self, batchSize = Params.BATCH_SIZE):
        train_ids, validation_ids = model_selection.train_test_split(self.Ids, random_state=42, test_size=0.30)
        return self.GenerateDataBatch(train_ids,batchSize),self.GenerateDataBatch(validation_ids,batchSize)

    def GetGrayImage(self,id):
        return self.images_gray[id]

    def GetOrginalImage(self,id):
        return imgOp.ImageToArrayConverter(os.path.join(self.imagesDir,DataAccess.GetFileName(id)))

    def GenerateDataBatch(self,data, batchSize):
        while True:
            resX = []
            resY = []
            ids = np.random.choice(data,replace=False,size=batchSize)
            for id in ids:
                image = self.images_gray[id]
                mask = self.masks[id]
                image, mask = DataAccess.GetModifiedData(image, mask)

                resX.append(image)
                resY.append(mask)

            yield np.asarray(resX, dtype=np.float32), np.asarray(resY, dtype=np.float32)

    def GetImageId(fileName):
        return fileName[:8]

    def GetFileName(fileId):
        return "{}{}".format(fileId,DataAccess.FileExtension)

    def GetModifiedData(image,mask,shift_limit=(-0.25, 0.25),rotate_limit=(-180, 180), borderMode="wrap", u=0.7):
        if np.random.random() < u:
            (height, width, chanels) = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])

            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)
            
            image = sciImg.shift(sciImg.rotate(image,angle,mode=borderMode,reshape =False),(dx,dy,0),mode=borderMode)
            mask = sciImg.shift(sciImg.rotate(mask,angle,mode=borderMode,reshape =False),(dx,dy,0),mode=borderMode)
        return image, mask