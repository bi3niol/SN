from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from keras.losses import binary_crossentropy
import keras.backend as K
import Params
import os
class NeuralNetwork(object):

    def __init__(self, modelPath=None):
        
        self.__model = self.__CreateModel()
        self.name = "Empty Name"
        if modelPath != None:
            self.name = os.path.basename(modelPath)
            self.__model.load_weights(modelPath)
        self.__model.compile(loss=NeuralNetwork.bce_dice_loss, optimizer=Adam(lr=1e-4), metrics=[NeuralNetwork.dice_coef])
        
        return super().__init__()
    def getModel(self):
        return self.__model;

    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
    
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - NeuralNetwork.dice_coef(y_true, y_pred)

    def bce_dice_loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + NeuralNetwork.dice_coef_loss(y_true, y_pred)

    def __CreateModel(self):
        inputs = Input(shape=(Params.IMG_WIDTH,Params.IMG_HEIGHT,1))

        down0a, down0a_res = self.__Down(24, inputs)
        down0, down0_res = self.__Down(64, down0a)
        down1, down1_res = self.__Down(128, down0)
        down2, down2_res = self.__Down(256, down1)
    
        center = Conv2D(256, (3, 3), padding='same')(down2)
        center = BatchNormalization(epsilon=1e-4)(center)
        center = Activation('relu')(center)
        center = Conv2D(256, (3, 3), padding='same')(center)
        center = BatchNormalization(epsilon=1e-4)(center)
        center = Activation('relu')(center)

        up2 = self.__Up(256, center, down2_res)
        up1 = self.__Up(128, up2, down1_res)
        up0 = self.__Up(64, up1, down0_res)
        up0a = self.__Up(24, up0, down0a_res)

        classify = Conv2D(1, (1, 1), activation='sigmoid', name='Output')(up0a)

        model = Model(inputs=inputs, outputs=classify)

        return model

    def __Down(self,filters, input_):
        down_ = Conv2D(filters, (3, 3), padding='same')(input_)
        down_ = BatchNormalization(epsilon=1e-4)(down_)
        down_ = Activation('relu')(down_)
        down_ = Conv2D(filters, (3, 3), padding='same')(down_)
        down_ = BatchNormalization(epsilon=1e-4)(down_)
        down_res = Activation('relu')(down_)
        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
        return down_pool, down_res

    def __Up(self,filters, input_, down_):
        up_ = UpSampling2D((2, 2))(input_)
        up_ = concatenate([down_, up_], axis=3)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        up_ = Conv2D(filters, (3, 3), padding='same')(up_)
        up_ = BatchNormalization(epsilon=1e-4)(up_)
        up_ = Activation('relu')(up_)
        return up_

    def PrintModel(self):
        print(self.__model.summary())

    def SaveModel(self,path):
        self.__model.save(path)

    def FitModel(self, trainDataGenerator, validationDataGenerator, epochs, validationSteps ,stepsPerEpoch,callbacks):

        return self.__model.fit_generator(trainDataGenerator, 
        steps_per_epoch = stepsPerEpoch, 
        epochs = epochs,
        callbacks = callbacks,
        verbose = 1,
        validation_data = validationDataGenerator,
        validation_steps = validationSteps)

    def Predict(self, x):
        print(x.shape)
        return self.__model.predict(x)