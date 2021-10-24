'''
This code was developed as part of the 16-720 (A) course
at Carnegie Mellon University.
'''
import numpy as np
import cv2
from util import refineF, _singularize
import scipy


'''
This function is used to find a corresponding point
in camera 2, for a given point in camera 1. 
Instead of a 2D search, epipolar line is computed
using F and linear (local) search is performed
'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    '''
    Args.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

    '''

    '''
    for a fundamental matrix F, F * [x1,y1]
    is the epipolar line
    '''
    epipolar_line = np.dot(F,np.array([x1,y1,
        1]).reshape(3,1))
    epipolar_line /= np.sqrt(epipolar_line[0]**2\
     + epipolar_line[1]**2)
    
    end_y,end_x = im2.shape[:2]
    
    '''
    In-order to calculate the similarity between two
    image patches we use a gaussian kernel. The kernel
    size of 17x17 is used as weight.
    Next few lines calculates a gaussian filter  
    '''
    filter_size = 17
    mean,sigma = 0,5
    '''
    Arrange [-1,1] points as 2D grid. Square of values
    indicate the distance of each point from center of filter. 
    '''
    points = np.meshgrid(np.linspace(-1,1,filter_size),
     np.linspace(-1,1,filter_size))
    
    g_filter = np.sqrt(points[0]**2 + points[1]**2)
    g_filter = np.exp(-(g_filter-mean)**2 / (2*sigma**2))
    
    g_filter /= g_filter.sum()
    '''
    Because image is 3D, stack filters together
    '''
    g_filter = np.dstack((g_filter,g_filter,g_filter))
    
    '''
    pre-compute the spliced (cropped) image 1.
    '''
    im1_window = im1[y1-int(filter_size/2):y1+int(filter_size/2)+1,\
    	x1-int(filter_size/2):x1+int(filter_size/2)+1,\
        :].astype(np.float32)
    
    '''
    Because images are almost similar, a local search in a 90 pixel
    window -45 to 45 will result in efficient search
    '''
    opt_x,opt_y = -1,-1
    opt_ssd = np.inf
    search_window = 45
    '''
    compute the range of y based on window, filter size
    '''
    y_min = y1 - search_window if y1 - search_window > \
    int(filter_size/2) else int(filter_size/2)
    
    y_max = y1 + search_window if y1 + search_window < \
    end_y - int(filter_size/2) else end_y - int(filter_size/2) - 2
    
    '''
    iterate from min y to max y
    '''
    for y in range(y_min,y_max):
        '''
        compute approx x coordinate using epipolar line
        and y coordinate
        '''
        x = int(-(epipolar_line[1] * y + \
            epipolar_line[2]) / epipolar_line[0])	
        
        '''
        Crop the region of interest (filter size x filter size) 
        '''
        im2_window = im2[y-int(filter_size/2):y+int(filter_size/2)+1,\
            x-int(filter_size/2):x+int(filter_size/2)+1,\
            :].astype(np.float32)
        
        '''
        compute SSD between the two crops
        weigh them using the gaussian kernel
        if weighted sum is least then coordinates
        are optimal
        '''
        diff_image = (im2_window - im1_window)**2
        diff_image *= g_filter
        g_ssd = diff_image.sum()
        if opt_ssd > g_ssd:
            opt_ssd = g_ssd
            opt_x = x
            opt_y = y
    return opt_x, opt_y