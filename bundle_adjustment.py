import numpy as np
import cv2
import scipy


'''
In order to write the code for rodrigues ans invRodrigues
I find the following link very useful:
https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
Hence this implementation is influenced by the above reference
'''
'''
This method computes the Rotation matrix (R) from a
given rotation vector (r) using the rodrigues formula
'''
def rodrigues(r):
    '''
    Args
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
    '''
    
    '''
    This tol is used to squash small angles to 0
    '''
    angle_tol = 1e-3
    '''
    Angle is the magnitude of vector r
    '''
    angle = np.sqrt(np.sum(r**2))
    if angle <= angle_tol:
    	R = np.eye(3,dtype = np.float32)
    	return R
    '''
    unit vector of r corresponds to the axis
    '''
    unit_r = r / angle
    '''
    compute hat{r}
    '''
    cross_r = np.array([[0,-unit_r[2],
        unit_r[1]],[unit_r[2],0,-unit_r[0]],
        [-unit_r[1],unit_r[0],0]])
    
    '''
    Rodrigues formula
    '''
    R = np.eye(3,dtype=np.float32) + \
    cross_r * np.sin(angle) + \
    np.linalg.matrix_power(cross_r,2)*(1-np.cos(angle))
    
    return R


'''
The following functions are implemented as part of
inverse rodrigues. Inverse rodrigues takes a rotation
matrix R and returns a corresponding rotation vector r.
'''
'''
half_s function flips the signs of r to ensure that
angles are limited to half hemisphere. This is to 
generate unique r.
'''
def half_s(x,t):
    x1,x2,x3 = x[0],x[1],x[2]
    '''
    ||r|| = pi and 
    ((r1 =r2 = 0 and r3<0) or
    (r1=0 and r2<0) or
    (r1=0))
    '''
    if np.abs(np.sqrt(np.sum(x**2))-np.pi) <= t and \
        ((np.abs(x1)<=t and np.abs(x2)<= t and x3<-t) or \
        (np.abs(x1)<=t and x2<-t) or (x1<-t)):
        return -x
    else:
        return x
'''
custom_arctan is used to generate proper
tan_inverse and handle corner cases
avoid rotation based similarity
'''
def custom_arctan(sin,cos,t):
	if cos > t:
		return np.arctan(sin/cos)
	elif cos < -t:
		return np.pi + np.arctan(sin/cos)
	elif np.abs(cos) < t and sin > t:
		return np.pi / 2
	elif np.abs(cos) < t and sin < -t:
		return -np.pi / 2

'''
inverse rodrigues function takes in a 
3x3 Rotation matrix and returns a rotation
vector r
'''
def invRodrigues(R):
    '''
    Args.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
    '''

    '''
    angle tol is used to squash small values
    of angle to zero
    '''
    angle_tol = 1e-3
    
    temp_matrix = (R - R.T) / 2.
    
    temp_elems = np.array([temp_matrix[2,1],
        temp_matrix[0,2],temp_matrix[1,0]])
    
    sin = np.sqrt(np.sum(temp_elems**2))
    '''
    sum of diagnol elements
    '''
    cos = (R[0,0]+R[1,1]+R[2,2] - 1) / 2.
    
    '''
    case when r is zero vector
    '''
    if np.abs(sin - 0) <= angle_tol and \
    np.abs(cos - 1) <= angle_tol:
        r = np.zeros(3,np.float32)
        return r
    
    elif np.abs(sin-0) <= angle_tol and \
    np.abs(cos + 1) <= angle_tol:
        tempV = R + np.eye(3)
        
        non_zero_col_idx = np.argwhere(
            np.sqrt(np.sum(tempV**2,axis=1)) > angle_tol)
        
        V = tempV[:,non_zero_col_idx[0,0]].T
        
        unit_u = V / np.sqrt(np.sum(V**2))
        
        r = half_s(unit_u*np.pi, angle_tol)
        return r
    
    elif np.abs(sin) > angle_tol:
        u = temp_elems / sin
        angle = custom_arctan(sin,cos,angle_tol)
        r = u * angle
        return r


'''
This function is the input to the optimizer.
It takes the inputs and computes the reprojection
error which is minimized by the optimizer
x is updated to x + delta_x by the optimizer
and consequently updates are calculated.
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    '''
    Args.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    '''

    '''
    first 3 elements are the rotation vector
    next 3 elements are the translation vector
    rest are the estimated 3D points
    '''
    r = x[:3].reshape(-1)
    t = x[3:6].reshape(-1)
    ws = x[6:].reshape(-1,3)
    '''
    homogenise the 3D points
    '''
    h_ws = np.hstack((ws,np.ones((ws.shape[0],1))))
    cam1_mat = np.dot(K1,M1)
    '''
    compute the camera matrix M2
    '''
    R = rodrigues(r)
    M2 = np.zeros((3,4))
    M2[:,:3] = R
    M2[:,-1] = t.reshape(-1,1).T
    cam2_mat = np.dot(K2,M2)
    '''
    calculate the reprojection error
    '''
    reproject_1 = np.dot(cam1_mat,h_ws.T)
    reproject_1 /= reproject_1[-1]
    reproject_1 = reproject_1[:-1].T
    
    reproject_2 = np.dot(cam2_mat,h_ws.T)
    reproject_2 /= reproject_2[-1]
    reproject_2 = reproject_2[:-1].T
    
    '''
    residuals contain the re-projection errors per point pair
    '''
    residuals = np.concatenate([(p1-reproject_1).reshape([-1]),
     (p2-reproject_2).reshape([-1])])

    return residuals


'''
bundle adjustment is a joint optimization method wherein 
multiple items are simultaneously optimized
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    '''
    Args.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    
    '''
    
    '''
    RANSAC estimate of R and T are used
    '''
    R_init = M2_init[:,:3]
    r_init = invRodrigues(R_init)
    t_init = M2_init[:,-1]
    '''
    initialization vector is formed. optimizer makes
    updates based on this vector
    '''
    initial_input = np.concatenate((r_init,t_init,
        P_init.flatten()))
    
    '''
    Code creates a lambda function which is given as
    input to the optimizer
    '''
    opt_func = lambda x: rodriguesResidual(K1, M1, p1,
     K2, p2, x)
    
    '''
    Scipy library uses the least sq optimizer to
    minimize the residual sum of squares
    '''
    opt_x,_ = scipy.optimize.leastsq(opt_func,
    initial_input,maxfev=2000)
    
    print(np.sum(opt_func(initial_input)**2))
    print(np.sum(opt_func(opt_x)**2))
    
    '''
    optimal camera matrix with minimal
    residual sum of squares is formed
    '''
    opt_r = opt_x[:3].reshape(-1)
    opt_t = opt_x[3:6].reshape(-1)
    depth_p = opt_x[6:].reshape(-1,3)

    opt_R = rodrigues(opt_r)
    opt_M2 = np.zeros((3,4))
    opt_M2[:,:3] =opt_R
    opt_M2[:,-1] = opt_t.reshape(-1,1).T

    return opt_M2, depth_p