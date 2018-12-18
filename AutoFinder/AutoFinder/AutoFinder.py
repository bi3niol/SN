import io
import ImageOperations as imgOp
import ProcessImagesToDataFile as cd
import numpy as np
import DataAccess as DA
import scipy.ndimage as sci
import Params
from keras.callbacks import EarlyStopping, ModelCheckpoint
from NeuralNetwork import NeuralNetwork

#sources
#https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9
#https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272
#https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
# Down -> Up
#https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_segmentation.html
#https://medium.com/weightsandbiases/car-image-segmentation-using-convolutional-neural-nets-7642448028f6
#
#https://blog.goodaudience.com/using-convolutional-neural-networks-for-image-segmentation-a-quick-intro-75bd68779225
#https://www.di.ens.fr/sierra/pdfs/tip10b.pdf
#https://github.com/malhotraa/carvana-image-masking-challenge/blob/master/notebooks/model.py
# budowa roznego rodzaju sieci
#https://machinelearningmastery.com/keras-functional-api-deep-learning/
#wyjasnienie CNN
#https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
#
#image opperations
#https://docs.scipy.org/doc/scipy/reference/ndimage.html
#
#restore best weights
#
#generowanie danych
#http://repository.supsi.ch/5145/1/IDSIA-04-12.pdf
if __name__ == "__main__":
    #img =
    #imgOp.To2DArrayFromImage(imgOp.ToGrayImage(imgOp.ImageToArrayConverter("./Data/images/Image_06.png")))
    #mask = imgOp.ImageToArrayConverter("./Data/labels/Image_06.png")
    #img, mask = DA.DataAccess.GetModifiedData(img,mask)
    #res = imgOp.AddMask(imgOp.ToGrayImageFrom2D(img),mask,1)
    #imgOp.ShowImage(res)

    da = DA.DataAccess()
    
    trainGen, testGen = da.GetDataGenerators()

    earlyStopping = EarlyStopping(monitor='val_loss', 
                                  patience=2, 
                                  verbose=1, 
                                  min_delta = 0.0001,
                                  mode='min',)

    modelCheckpoint = ModelCheckpoint(Params.MODEL_CHECKPOINT,
                                      monitor = 'val_loss', 
                                      save_best_only = True, 
                                      mode = 'min', 
                                      verbose = 1,
                                      save_weights_only = True)

    nn = NeuralNetwork(Params.MODEL_LOCATION)

    for x in range(23,27):
        id = "Image_{}".format(x)
        gray = da.GetGrayImage(id)
        orginal = da.GetOrginalImage(id)
        gray, orginal = DA.DataAccess.GetModifiedData(gray,orginal)
        y = nn.Predict(np.array([gray]))
        res = imgOp.AddMask(orginal ,y[0],1)
        imgOp.ShowImage(res)

    #nn.PrintModel()
    #nn.FitModel(trainGen,testGen,200,2,10,[modelCheckpoint])
