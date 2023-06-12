from skimage import io
from scipy import fft
from scipy import signal
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

def grayscale_bit_depth(grayscale, bit_depth):
    return (2**bit_depth-1)*(grayscale - grayscale.min())/(grayscale.max() - grayscale.min())

def scan_image(img1, img2, window_size):
    height1, width1 = np.shape(img1)
    height2, width2 = np.shape(img2)
    if(height1 % window_size > 0 or height2 % window_size>0 or width1 % window_size>0 or width2 % window_size >0):
        raise ValueError('Window Size Doesn\'t Divide Evenly into Image Coordinates')
    elif(height1!=height2 or width1!=width2):
        #print(f'height1 {height1} height2 {height2}')
        raise ValueError('First and Second Images have different sizes')
        
    magnitude_matrix = np.zeros((int(height1/window_size), int(width1/window_size)))
    displacement_matrix = np.zeros((int(height1/window_size), int(width1/window_size), 2))
    znccs = np.zeros((int(height1/window_size), int(width1/window_size)))
                         
    for row in np.arange(0, np.shape(magnitude_matrix)[0],1):
        for column in np.arange(0, np.shape(magnitude_matrix)[1],1):
            #print(f'cross correlating img1[{row*window_size}:{window_size*(row+1)},{column*window_size}:{window_size*(column+1)}] with img2[{row*window_size}:{window_size*(row+1)},{column*window_size}:{window_size*(column+1)}]')
            max_coords = find_max_2d(cross_correlate(img1[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)], 
                                                                    img2[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)]))
            magnitude_matrix[row, column] = np.hypot(max_coords[0], max_coords[1])
            displacement_matrix[row, column] = (max_coords[0], max_coords[1])
            znccs[row, column], _ = stats.pearsonr(img1[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)].flatten(), 
                                                         img2[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)].flatten())
    return (displacement_matrix, magnitude_matrix, znccs)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def cross_correlate(img1, img2):
    #expects square matrix
    if(len(img1[0])!=len(img2[0]) or len(img1[:0])!=len(img2[:0]) or np.shape(img1)[0]!= np.shape(img1)[1] or np.shape(img2)[0]!= np.shape(img2)[1]):
        raise ValueError('Images must be square matrices of equal length')
    else:
        img1_autocorr = signal.convolve(img1, img1)
        img2_autocorr = signal.convolve(img2, img2)
        avg_autocorr = 0.5*(img1_autocorr + img2_autocorr)
        cross_correlation = np.flip(signal.convolve(img1, np.flip(img2))) #cross-correlate two images 
        cc_normed = cross_correlation - avg_autocorr #subtract out the average of each image's autocorrelation to get cleaner peaks
        cc_normed = (cc_normed - np.min(cc_normed))/(np.max(cc_normed) - np.min(cc_normed)) #scale from 0 to 1
        return cc_normed
    
def find_max_2d(matrix):
    # height = np.shape(matrix)[0]
    # width = np.shape(matrix)[1]
    # #print(f'height {height} width {width}')
    # max_coords = [0,0]
    # for x in np.arange(0,width,1):
    #     for y in np.arange(0,height,1):
    #         #print(f'max_coords: {max_coords}')
    #         if(matrix[y,x]>matrix[max_coords[1], max_coords[0]]):
    #             max_coords = [x, y]
    # return max_coords
    dcc_dx = np.zeros(matrix.shape)
    dcc_dy = np.zeros(matrix.shape)
    dcc_dx[:,1:] = matrix[:,1:] - matrix[:,:-1]
    dcc_dy[1:,:] = matrix[1:,:] - matrix[:-1,:]
    derivative_matrix = np.sqrt(dcc_dx**2 + dcc_dy**2)
    [y,x] = [np.where(derivative_matrix==np.max(derivative_matrix))[i][0] for i in [0,1]] #return coordinates for maximum magnitude of change, this should be roughly centered on the inflection point 
    #print(f' Matrix shape : {matrix.shape} UV coords:({x-len(matrix[0])/2}, {y-len(matrix[0,:])/2}) angle: {np.arctan((y-len(matrix[0,:])/2)/(x-len(matrix[0])/2)) * 180/np.pi}')
    return [x-len(matrix[0])/2,y-len(matrix[0,:])/2]
        