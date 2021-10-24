# 3D reconstuction using stereo Correspondences

This repo contains code for 3D reconstruction given correspondence between two views of a scene. The approach first computes the fundamental and Essential Matrix which relate the two cameras. Then with first camera as basis, we estimate the relative values of the second camera matrix. 

Next we can triangulate the two corresponding points from stereo-images to generate 3D reconstruction.


Repo Structure:
* **fundamental_matrix**: This script contains implementations for the following:
   * Eight Point Algorithm for Fundamental Matrix computation (given list of corresponding points)
   * Essential Matrix from Fundamental Matrix and camera Intrisic Parameters
   * Ransac Implementation for robust Fundamental Matrix computation

* **epipolar_corresondence**: Code computing the matching point (x2,y2) in right image, given the Fundamental Matrix and point (x1,y1) in left image

* **bundle_adjustment**: In case of noisy correspondences, code for jointly optimizing the optimal Camera Extrisic Matrix and Estimated Depth using bundle adjustment.
    * Bundle Adjustment optimizes the reprojection error
* **run_bundle**: Code for running the bundle adjustment for producing estimated camera matrix and depths. 


Assumptions:
1. The code assumes that point correspondence is already computed for Fundamental Matrix
2. The code assumes that intrisic matrix for cameras is already known

_______
#### Notes
Major part of the code was developed as part of 16720 Intro to CV course at Carnegie Mellon University. This repo is under development.

For Bundle Adjustment, the repo borrows from the descriptions in https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf.

To contact the author, feel free to write to paritosm@andrew.cmu.edu