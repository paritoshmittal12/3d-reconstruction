import cv2
import numpy as np

from fundamental_matrix import eightpoint, essentialMatrix
from fundamental_matrix import triangulate, ransacF

from bundle_adjustment import rodrigues, invRodrigues, bundleAdjustment


from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

'''
Assumption, 
(1) The code assumes we have a noisy set of correspondences between left and 
right images: noisy_pts1, noisy_pts2

(2) The code assumes we know the intrisics of left (cam1K) and right (cam2K) 
'''


# img1 = cv2.imread('../data/im1.png')[:,:,::-1]
# img2 = cv2.imread('../data/im2.png')[:,:,::-1]

M = np.max(img1.shape)

'''
Use RANSAC + EightPoint to get Fundamental Matrix
'''
robust_F, inlier_idx = ransacF(noisy_pts1,noisy_pts2,M,nIters=300,tol=1.5)

pts1,pts2 = noisy_pts1[inlier_idx], noisy_pts2[inlier_idx]


'''
Commpute Essential Matrix
'''
E = essentialMatrix(robust_F,cam1K,cam2K)

'''
We can extract the Camera Matrix for second camera using E.
Assume Camera1 has Identity Matrix
'''

M1_cand = np.zeros((3,4))
M1_cand[:,:3] = np.eye(3)
C1_cand = np.dot(cam1K,M1_cand)

'''
Four Camera Matrix are candidates. 
We select the Camera Matrix with minimum # of 3D points
having z<0.
'''
min_neg_Z = pts1.shape[0]+1
err_max = -1
M2_opt = None
w_points_opt = None
C2_opt = None

for i in range(M2s.shape[2]):
	M2_cand = M2s[:,:,i]
	C2_cand = np.dot(cam2K,M2_cand)
	w_points, err = triangulate(C1_cand,pts1,C2_cand,pts2)
	count_neg_Z = np.sum(w_points[:,2]<0)
	
	if count_neg_Z ==0 and err > err_max:
	
		min_neg_Z = count_neg_Z
		M2_opt = M2_cand
		C2_opt = C2_cand
		w_points_opt = w_points
		err_max = err

		
depth_array,_ = triangulate(C1_cand,pts1,C2_opt,pts2)

opt_M2, depth_points = bundleAdjustment(cam1K, M1_cand, pts1, cam2K, M2_opt, pts2, depth_array)

