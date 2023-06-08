import numpy as np
from skimage import io
from Analysis_Tools import *

class ImageStack():
    def __init__(self, path, bit_depth=32):
        self.imgs = np.array([grayscale_bit_depth(rgb2gray(x), bit_depth) for x in io.imread(path)])
        self.shape = self.imgs.shape
        
    def __getitem__(self, index):
        return self.imgs[index]
    
    def shape(self):
        return np.shape(self.imgs)