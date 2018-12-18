import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def ToGrayImage(image):
    outImg = (3 * image[:,:,0] + 5 * image[:,:,1] + image[:,:,2]) / 9
    return outImg.reshape([outImg.shape[0],outImg.shape[1],1])

def ToGrayImageFrom2D(image):
    outImg = np.zeros([image.shape[0],image.shape[1],3])
    outImg[:,:,0] = image
    outImg[:,:,1] = image
    outImg[:,:,2] = image
    return outImg

def To2DArrayFromImage(image):
    return image[:,:,0]

def AddMask(image,mask,chanel, maskalfa=0.3):
    res = np.zeros(image.shape)
    res[:,:,:] = image[:,:,:]
    res[:,:,chanel] = (image[:,:,chanel] + maskalfa*mask[:,:,0])
    return res

def ImageToArrayConverter(imagePath):
    image = mpimg.imread(imagePath)
    return np.array(image)

def ShowImage(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()