'''
This code was developed as part of the 16-720 (A) course
at Carnegie Mellon University.
'''

import numpy as np
import cv2
import scipy

'''
The following code implements an eight point
algorithm to estimate Fundamental Matrix F
'''
def eightpoint(pts1, pts2, M):
    '''
    Args
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
    '''

    '''
    This transformation matrix is used to scale and
    un-scale the points - computed F
    '''
    
    transformation_matrix = np.array([[1./M,0,0],
        [0,1./M,0],[0,0,1]])
    
    '''
    scale the data
    '''
    pts1 = pts1/M
    pts2 = pts2/M

    N = pts1.shape[0]
    A = np.ones((N,9))
    
    '''
    A is an Nx9 matrix wherein each point pair i
    corresponds to one row 
    '''
    for i in range(0,N):
        x,y = pts1[i]
        xp,yp = pts2[i]

        A[i,0] = xp * x
        A[i,1] = xp * y
        A[i,2] = xp * 1
        A[i,3] = yp * x
        A[i,4] = yp * y
        A[i,5] = yp * 1
        A[i,6] = 1 * x
        A[i,7] = 1 * y

    '''
    To estimate F, we take the last 
    column of V where U, E, V^T = svd(A)
    This corresponds to least sq solution
    of AF = 0.
    '''
    U, E, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape((3,3))
    
    '''
    Can perform local optimization for improved F matrix
    '''

    '''
    Unscale F so as to work with normal unscaled
    coordinates
    '''
    unscaledF = np.dot(transformation_matrix.T,
        np.dot(F,transformation_matrix))
    return unscaledF

'''
Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    
    # F = k2-TEK1-1
    # FK1 = K2-TE
    # K2TFK1 = E
    
    return np.dot(K2.T,np.dot(F,K1))

'''
Triangulate a set of 2D coordinates in the image to a set
         of 3D points.
    
'''
'''
This function calculates a 3D estimate of point
using two corresponding points and their camera matrices
'''
def triangulate(C1, pts1, C2, pts2):
    '''
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
    
    '''
    N = pts1.shape[0]
    
    '''
    initialize array
    '''
    w_points = np.zeros((N,3))
    reprojection_error = 0

    for i in range(N):
        '''
        For each point in the corresponding point pairs
        an A matrix is computed
        '''
        x1,y1 = pts1[i]
        x2,y2 = pts2[i]
        cam1i = np.array([[-1,0,pts1[i,0]],[0,-1,pts1[i,1]]])
        cam2i = np.array([[-1,0,pts2[i,0]],[0,-1,pts2[i,1]]])
        '''
        A is a 4x4 matrix wherein 2 rows come from C1, x1,y1
        and 2 rows come from C2, x2, y2
        '''
        A = np.vstack((np.dot(cam1i,C1),np.dot(cam2i,C2)))
        '''
        Estimated 3D point is the least square solution of
        Aw = 0. 
        '''
        U,E,VT = np.linalg.svd(A)
        w_i = VT[-1]
        '''
        Normalize w_i
        '''
        w_i /= w_i[-1]
        '''
        Reprojection error is calculated by 
        re projecting the 3D point back into 2D camera
        plane and estimating the squared error.
        '''
        w_points[i] = w_i[:3]
        reproject_1 = np.dot(C1,w_i)
        reproject_1 /= reproject_1[2]
        
        reproject_2 = np.dot(C2,w_i)
        reproject_2 /= reproject_2[2]
        
        reprojection_error += np.sum((pts1[i]\
            -reproject_1[:2])**2)
        reprojection_error += np.sum((pts2[i]\
            -reproject_2[:2])**2)
        
    return w_points, reprojection_error


'''
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''


'''
The following function defines RANSAC
for fundamental matrix computation
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    '''
    Args.

    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
    '''
    
    N = len(pts1)
    '''
    stack points with ones to make them homogenous
    '''
    pts1 = np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2 = np.hstack((pts2,np.ones((pts2.shape[0],1))))
    max_inliers = -1
    opt_F = None
    max_inlier_idx = None
    
    '''
    iterate over all iterations
    '''
    for i in range(nIters):
        current_inliers = 0
        
        '''
        randomly sample 8 points (minimum for algo)
        '''
        random_idx = np.random.randint(0,N,8)

        temp_F = eightpoint(pts1[random_idx,:2]
            ,pts2[random_idx,:2],M)
        '''
        Calculate the epipolar line for each
        point in img 1.
        '''
        pred_lines = np.dot(temp_F,pts1.T)
        pred_lines = pred_lines / (np.sqrt(pred_lines[0,:]**2\
         + pred_lines[1,:]**2))

        '''
        compute the distance of each point in pts2 with its 
        corresponding epipolar line. This metric is used to
        measure inliers
        '''
        line_dist = np.abs(np.sum(pts2.T * pred_lines,axis=0))

        '''
        compute inliers
        '''
        inliers = line_dist < tol
        current_inliers = inliers.sum()
        if max_inliers < current_inliers:
            max_inliers = current_inliers
            max_inlier_idx = np.argwhere(inliers==True)
    
    '''
    refine the estimate over all inliers 
    '''
    # print(max_inliers)
    final_F = eightpoint(pts1[max_inlier_idx[:,0],:2],
        pts2[max_inlier_idx[:,0],:2],M)

    return final_F, max_inlier_idx[:,0]