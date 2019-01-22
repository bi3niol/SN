import io
import os
import ImageOperations as imgOp
import ProcessImagesToDataFile as cd
import numpy as np
import DataAccess as DA
import scipy.ndimage as sci
import Params
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from NeuralNetwork import NeuralNetwork
import json
from keras.utils import plot_model
from Comitet import Comitet
import matplotlib.pyplot as plt
import pandas as pd

#sources
#https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9
#https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272
#https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
# Down -> Up
#https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html
#https://medium.com/weightsandbiases/car-image-segmentation-using-convolutional-neural-nets-7642448028f6
#https://blog.goodaudience.com/using-convolutional-neural-networks-for-image-segmentation-a-quick-intro-75bd68779225
#https://www.di.ens.fr/sierra/pdfs/tip10b.pdf
# budowa roznego rodzaju sieci
#https://machinelearningmastery.com/keras-functional-api-deep-learning/
#wyjasnienie CNN
#https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
#image opperations
#https://docs.scipy.org/doc/scipy/reference/ndimage.html
#generowanie danych
#http://repository.supsi.ch/5145/1/IDSIA-04-12.pdf
class History(Callback):
    def __init__(self, historyPath,epochPath, loadData):
        self.historyPath = historyPath
        self.epochPath = epochPath
        self.loadData = loadData
        return super().__init__()

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

        if not self.loadData:
            return

        if os.path.exists(self.historyPath):
            with open(self.historyPath,"r") as f:
                self.history = json.load(f)

        if os.path.exists(self.epochPath):
            with open(self.epochPath,"r") as f:
                self.epoch = json.load(f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.historyPath != None:
            with open(self.historyPath,"w") as f:
                json.dump(self.history,f)

        if self.epochPath != None:
            with open(self.epochPath,"w") as f:
                json.dump(self.epoch,f)


def GetPredictData(id, gray, orginal, mask, y_pred, title=None):
    res = imgOp.AddMask(orginal, y_pred, 1)
    if not (title is None):
        plt.title(title)
    acc = Accuracy(mask,y_pred, Params.PREDICT_DATA_THRESHOLD)
    plt.text(0,0,"Dokladnosc : {0:.2f}".format(acc),horizontalalignment='left',verticalalignment='top',bbox=dict(facecolor='red'))
    imgOp.ShowImage(res)

def Accuracy(y, y_pred, threshold = None):
    if not threshold is None:
        y_pred[y_pred>=threshold] = 1
        y_pred[y_pred<threshold] = 0

    return (2 * ((y * y_pred).sum())) / (y.sum() + y_pred.sum() + 1)

def GetTestData(dataAccess):
     for x in Params.TEST_IMAGE_IDS:
        id = "Image_{}".format(x)
        gray = dataAccess.GetGrayImage(id, True)
        orginal = dataAccess.GetOrginalImage(id, True)
        mask = dataAccess.GetMask(id, True)
        yield id, gray, orginal, mask

def AddEmptyArrayToDictionaryIfNeed(dict,key):
    if not key in dict:
        dict[key] = []


if __name__ == "__main__":
    da = DA.DataAccess()
    
    trainGen, testGen = da.GetDataGenerators()

    earlyStopping = EarlyStopping(monitor='val_loss', 
                                  patience=20, 
                                  verbose=1, 
                                  min_delta = 0.0001,
                                  restore_best_weights = True,
                                  mode='min',)

    modelCheckpoint = ModelCheckpoint(Params.MODEL_CHECKPOINT,
                                      monitor = 'val_loss', 
                                      save_best_only = True, 
                                      mode = 'min', 
                                      verbose = 1,
                                      save_weights_only = True)
    history = History(Params.HISTORY_PATH, Params.EPOCH_PATH, True)

    if Params.CREATE_MODEL:
        nn = NeuralNetwork(Params.MODEL_LOCATION)

    if Params.PLOT_MODEL:
        plot_model(nn.getModel(),to_filte='model.png')

    if Params.PRINT_MODEL:
        nn.PrintModel()
    if Params.TEST_MODEL:
        for id, gray, org, mask in GetTestData(da):
            y = nn.Predict(np.array([gray]))
            GetPredictData(id, gray, org, mask,y[0],"{}-{}".format(nn.name,id))

    if Params.TRAIN_MODEL:
        nn.FitModel(trainGen,testGen,200,2,10,[modelCheckpoint, history, earlyStopping])

    if Params.CREATE_COMITET:
        models = []
        for modelPath in Params.COMITET_MODELS:
            models.append(NeuralNetwork(modelPath))
        comitet = Comitet(models, Params.THRESHOLD)

    if Params.COMITET_PREDICT:
        for id, gray, org, mask in GetTestData(da):
            y = comitet.Predict(gray)
            GetPredictData(id, gray, org, mask,y,"Comitet-Threshold {}-{}".format(Params.THRESHOLD,id))

    if Params.TEST_MODELS_IN_COMITET:
        data = { 
            "Model" : []
            }
        for model in models:
            data["Model"].append(model.name)
            for id, gray, org, mask in GetTestData(da):
                y = model.Predict(np.array([gray]))
                acc = Accuracy(mask, y[0],Params.THRESHOLD)
                AddEmptyArrayToDictionaryIfNeed(data,id)
                data[id].append(acc)

        df = pd.DataFrame(data)
        df.to_csv(Params.TEST_MODELS_IN_COMITET_OUT_FILE)

    if Params.CREATE_THRESHOLD_TEST_FOR_COMITET:
        data = { 
            "Thresholds" : []
            }
        for threshold in Params.COMITET_THRESHOLDS:
            data["Thresholds"].append(threshold)
            comitet.threshold = threshold
            for id, gray, org, mask in GetTestData(da):
                y = comitet.Predict(gray)
                acc = Accuracy(mask, y)
                AddEmptyArrayToDictionaryIfNeed(data,id)
                data[id].append(acc)

        df = pd.DataFrame(data)
        df.to_csv(Params.THRESHOLD_TEST_FOR_COMITET_OUT_FILE)
