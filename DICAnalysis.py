from Analysis_Tools import *
import numpy as np

class DICAnalysis():
    def __init__(self, ImageStack, correlation_window_size):
        self.images = ImageStack
        (self.n, self.height, self.width) = self.images.shape
        self.window_size = correlation_window_size #square subwindow size for cross-correlation
        #print(f'height: {height} width {width}')
        
    def Analyze(self):
        print('------Analyzing----------')
        height = self.height
        width = self.width
        n = self.n
        window_size = self.window_size
        imgs = self.images
        magnitude_stack = np.zeros((n, int(height), int(width)))
        displacement_stack = np.zeros((n, int(height), int(width), 2))
        zncc_stack = np.zeros((n, int(height), int(width)))
        
        for p in range(1,n):
            print(f'loop {p} of {n}')
            magnitude_map = np.zeros((int(height), int(width)))
            displacement_map = np.zeros((int(height), int(width),2))
            zncc_map = np.zeros((int(height), int(width)))
            displacements, magnitudes, znccs = scan_image(imgs[p-1], imgs[p], window_size)
            
            for y in range(0,height, window_size):
                for x in range(0,width, window_size):
                    magnitude_map[y:y+window_size,x:x+window_size] = np.ones((window_size, window_size)) * magnitudes[int(y/window_size),int(x/window_size)]
                    displacement_map[y+int(window_size/2),x+int(window_size/2)] = displacements[int(y/window_size),int(x/window_size)] #create sparse matrix same size as original image and center displacement vector in window 
                    zncc_map[y:y+window_size,x:x+window_size] = np.ones((window_size, window_size)) * znccs[int(y/window_size),int(x/window_size)]
            magnitude_stack[p] = magnitude_map
            displacement_stack[p] = displacement_map
            zncc_stack[p] = zncc_map
        return (displacement_stack, magnitude_stack, zncc_stack)
        # plt.imshow(imgs[p], cmap='Greys', origin='lower')
        # plt.imshow(cc_matrices, cmap='jet', alpha=0.25, origin='lower')
        # plt.colorbar()
        # plt.savefig(f'C:/users/ryanp/downloads/DIC/dic{p}.png')
        # plt.clf()