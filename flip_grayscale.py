from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class imageProcess(object):

    def __init__(self):
        """
        This is a simple image processing class 
        supporting conversion to grayscale and 
        flip horizontally/vertically
        
        r, g, b are weights for weighted average
        conversion to grayscale
        """
        self.r = 0.3
        self.g = 0.59
        self.b = 0.11

    def average(self, pixel):
        """
        Average of 3 channels
        """
        return (pixel[0] + pixel[1] + pixel[2]) / 3

    def weightedAverage(self, pixel):
        """
        Weighted gray scale based on human 
        eye perception
        """
        return self.r * pixel[0] + self.g * pixel[1] + self.b * pixel[2]

    def toGrayScale(self, flag, image):
        """
        Returns a grayscaled image and 
        saves it
        """
        assert (image.ndim == 3), "Input image is not RGB"
        grey = np.zeros((image.shape[0], image.shape[1]))  
        if flag == 0:
            for row in range(len(image)):
                for col in range(len(image[row])):
                    grey[row][col] = self.weightedAverage(image[row][col])
        if flag == 1:
            for row in range(len(image)):
                for col in range(len(image[row])):
                    grey[row][col] = self.average(image[row][col])
        
        img = Image.fromarray(grey)                    
        misc.imsave('gray.jpg', img)
        return img
    
    def flipHelper(self, flag, img, image):
        """
        Core of flipping functionality handling 
        pixel manipulations
        """
        rows = image.shape[0]
        cols = image.shape[1]
        if flag == 0:
            for row in range(len(image)):
                for col in range(len(image[row])):
                    img[rows-(row+1)][col] = image[row][col]
        if flag == 1:        
            for row in range(len(image)):
                for col in range(len(image[row])):
                    img[row][cols-(col+1)] = image[row][col]
        return img
    
    def flip(self, flag, image):
        """
        Returns flipped image vertically (flag=0) or 
        horizontally (flag=1) and saves it
        """
        if(image.ndim==3):
            img = np.zeros([image.shape[0],image.shape[1],image.ndim], dtype=np.uint8)
            img = self.flipHelper(flag, img, image)
            img = Image.fromarray(img, 'RGB')
            
        else:
            img = np.zeros((image.shape[0], image.shape[1])) 
            img = self.flipHelper(flag, img, image)
            img = Image.fromarray(img, 'gray')
                    
        misc.imsave('transformed.jpg', img)
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
    gray = process.toGrayScale(1, img)
    plt.imshow(gray)
    plt.show()    

    
    #Uncomment to flip
    imgFlip = process.flip(1, img)
    plt.imshow(imgFlip)
    plt.show()    
