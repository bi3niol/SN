import ImageOperations as imgOp
import os
import pickle

def CreateData(inputDir,outFile,imageProcessor):
    res = []
    for x in os.listdir(inputDir):
        filepath = os.path.join(inputDir,x)
        print(filepath)
        if os.path.isfile(filepath):
            res.append(imageProcessor(imgOp.ImageToArrayConverter(filepath)))
    with open(outFile, 'wb') as filehandle:
        pickle.dump(res,filehandle)
    return res

def LoadData(datafile):
    with open(datafile, 'rb') as filehandle:
       return pickle.load(filehandle)