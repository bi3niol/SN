import numpy as np

class Comitet(object):
    """description of class"""
    def __init__(self, models, threshold):
        self.__models = models
        self.threshold = threshold
        return super().__init__()

    def Predict(self, x):
        res = None

        for model in self.__models:
            tmp = model.Predict(np.array([x]))
            tmp = tmp[0]
            if res is None:
                res = tmp
            else:
                res+=tmp

        res = res/len(self.__models)
        res[res >= self.threshold] = 1
        res[res < self.threshold] = 0

        return res