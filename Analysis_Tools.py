from skimage import io
from scipy import fft
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def getAverage(img, u, v, n):
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += img[u + i][v + j]
    return float(s) / (2 * n + 1) ** 2

def getStandardDeviation(img, u, v, n):
    s = 0
    avg = getAverage(img, u, v, n)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += (img[u + i][v + j] - avg) ** 2
    return (s ** 0.5) / (2 * n + 1)

def zncc(img1, img2, u1, v1, u2, v2, n):
    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
    stdDeviation2 = getStandardDeviation(img2, u2, v2, n)
    avg1 = getAverage(img1, u1, v1, n)
    avg2 = getAverage(img2, u2, v2, n)

    s = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            s += (img1[u1 + i][v1 + j] - avg1) * (img2[u2 + i][v2 + j] - avg2)
    return float(s) / ((2 * n + 1) ** 2 * stdDeviation1 * stdDeviation2)

def grayscale_bit_depth(grayscale, bit_depth):
    return (2**bit_depth-1)*(grayscale - grayscale.min())/(grayscale.max() - grayscale.min())

def scan_image(img1, img2, window_size, magnitude=True):
    height1, width1 = np.shape(img1)
    height2, width2 = np.shape(img2)
    if(height1 % window_size > 0 or height2 % window_size>0 or width1 % window_size>0 or width2 % window_size >0):
        raise ValueError('Window Size Doesn\'t Divide Evenly into Image Coordinates')
    elif(height1!=height2 or width1!=width2):
        #print(f'height1 {height1} height2 {height2}')
        raise ValueError('First and Second Images have different sizes')
    if(magnitude):
        cc_matrix = np.zeros((int(height1/window_size), int(width1/window_size)))
    else:
        cc_matrix = np.zeros((int(height1/window_size), int(width1/window_size), 2))

    for row in np.arange(0, np.shape(cc_matrix)[0],1):
        for column in np.arange(0, np.shape(cc_matrix)[1],1):
            max_coords = find_max_2d(cross_correlate(img1[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)], 
                                                                    img2[row*window_size:window_size*(row+1),column*window_size:window_size*(column+1)]))
            if(magnitude):#print(f'max coords: {max_coords}')
                cc_matrix[row, column] = np.hypot(max_coords[0], max_coords[1])
            else:
                cc_matrix[row, column] = (max_coords[0], max_coords[1])
              
    return cc_matrix

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

        return np.flip(signal.convolve(img1, np.flip(img2))) - avg_autocorr
    
def find_max_2d(matrix):
    height = np.shape(matrix)[0]
    width = np.shape(matrix)[1]
    #print(f'height {height} width {width}')
    max_coords = [0,0]
    for x in np.arange(0,width,1):
        for y in np.arange(0,height,1):
            #print(f'max_coords: {max_coords}')
            if(matrix[y,x]>matrix[max_coords[1], max_coords[0]]):
                max_coords = [x-width, y-height]
    return max_coords
        