from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import logging

logging.basicConfig(filename="imageProcess.log", level=logging.DEBUG)
logger = logging.getLogger('imageProcess')


class imageProcess(object):

    def __init__(self, r = 0.3, g = 0.59, b = 0.11):
        """
        This is a simple image processing class 
        supporting conversion to grayscale and 
        flip horizontally/vertically
        
        r, g, b are weights for weighted average
        conversion to grayscale
        """
        self.r = r
        self.g = g
        self.b = b

    def __toGrayScaleHelper(self, image, flag):
        """
        Core of toGrayScale function involving pixel
        manipulation
            
        #Returns
            Grayscale image
        """
        logger.debug("No of dimension of input image is {}".format(image.ndim))
        assert (image.ndim == 3), "Input image is not RGB"
        
        #declare a numpy array that would store the grayscale
        grey = np.zeros((image.shape[0], image.shape[1]))  
        
        if flag == 0:
            grey = self.r * image[:,:,0] + self.g * image[:,:,1] 
            + self.b * image[:,:,2]

        if flag == 1:
            #using numpy array automatically handles overflow, 
            #preserving image quality
            grey = image.sum(axis=2)/3
            
        #returning datatype as "Image" instead of numpy array
        #because doing a plot.show on gray gives color distortion
        img = Image.fromarray(grey)                   
        return img
        
    def toGrayScale(self, image, save, filename = None,
                    flag=0):
        """
        Converts an RGB image to a grayscaled image and 
        saves it with filename
           
        #Arguments
            image: input image to be converted
            flag: specifies whether averaging (1)
                or weighted averaging desired (0)
                0 is set as default
            save: whether to save(1) or not
            filename: name with which the transformed
            image is stored
            
        #Returns
            Grayscale image
        """
        logger.info("Conversion to grayscale and saving as {}".format(filename))
        
        img = self.__toGrayScaleHelper(image, flag)
        
        if save == True:
            assert(filename != None), "Filename not specified"
            misc.imsave(filename, img)
            
        return img
    
    def ___flipHelper(self, flag, image):
        """
        Core of flipping functionality handling 
        pixel manipulations
        
        #Returns
            flipped image horizontally/vertically
        """
        if flag==0:
            img = np.flipud(image)
            
        if flag==1:
            img = np.fliplr(image)
                    
        return img
    
    def flip(self, image, save, filename=None, flag=0):
        """
        Flips image vertically/horizontally
        and saves it
        
        #Arguments
            image: input image to be converted
            flag: if flag 0 then flipped vertically
            and if 1 then horizontally
            save: whether to save(True) or not
            filename: name with which the transformed
            image is stored
            
        #Returns
            transformed image
        """
        
        logger.info("Flipping image and saving as {}".format(filename))
        
        img = self.___flipHelper(flag, image)
        
        if save == True:
            assert(filename != None), "Filename not specified"
            misc.imsave(filename, img)
        
        return img


    def showImage(self, filename):
        """
        Display an image file from filename
        """
        image = misc.imread(filename)
        plt.imshow(image)
        plt.show()
        


if __name__ == '__main__':

    filename = 'lenna.jpg'
    img = misc.imread(filename)
    
    #Create object instance
    process = imageProcess()
    process.showImage(filename)
    
    #Uncomment to turn to grayscale
    gray = process.toGrayScale(img, True, 'gray1.jpg')
    plt.imshow(gray)
    plt.show()    

    
    #Uncomment to flip
    imgFlip = process.flip(img, True, 'flip.jpg')
    plt.imshow(imgFlip)
    plt.show()    
    
