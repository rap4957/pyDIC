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
        cc_stack = np.zeros((n, int(height), int(width)))
        zncc_stack = np.zeros((n, int(height), int(width)))
        
        for p in range(1,n):
            print(f'loop {p} of {n}')
            cc_matrices = np.zeros((int(height), int(width)))
            znccs = np.zeros((int(height), int(width)))
            magnitudes = scan_image(imgs[p-1], imgs[p], window_size)
            velocities = scan_image(imgs[p-1], imgs[p], window_size, magnitude=False)
            for y in range(0,height, window_size):
                for x in range(0,width, window_size):
                    cc_matrices[y:y+window_size,x:x+window_size] = np.ones((window_size, window_size)) * magnitudes[int(y/window_size),int(x/window_size)]
                    znccs[y:y+window_size,x:x+window_size] = np.ones((window_size, window_size))
            cc_stack[p] = cc_matrices
            zncc_stack[p] = znccs
        return cc_stack
        # plt.imshow(imgs[p], cmap='Greys', origin='lower')
        # plt.imshow(cc_matrices, cmap='jet', alpha=0.25, origin='lower')
        # plt.colorbar()
        # plt.savefig(f'C:/users/ryanp/downloads/DIC/dic{p}.png')
        # plt.clf()