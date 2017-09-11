# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:00:42 2017





Visual Studio
@author: Neda
"""
#reset

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy

import h5py #load mat file	

#from skimage import exposure
from sklearn.ensemble import RandomForestRegressor
#from sklearn.multioutput import MultiOutputRegressor
#from numpy import unravel_index
import scipy.ndimage as ndimage
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import KFold
import time
from sklearn import cross_validation
import sklearn

#from sklearn import decomposition




##from skimage.measure import structural_similarity as ssim

#import cv2
#from skimage.feature import greycomatrix, greycoprops
#from matplotlib  import cm

#import scipy.io as sio
'''
#Mean Squared Error (MSE) and the Structural Similarity Index (SSIM) functions.
While the MSE is substantially faster to compute, it has the major drawback of (1) being applied globally and (2) only estimating the perceived errors of the image.
On the other hand, SSIM, while slower, is able to perceive the change in structural information of the image by comparing local regions of the image instead of globally.
'''

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	#s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f" % (m))
      #plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()



my_path = os.path.abspath(__file__)
start_t = time.time()
print "loading signals..."

#signal_file = h5py.File('C:\\Users\\Neda\\Desktop\\Neda\\from_luis\\signal_10.mat','r')
signal_file = h5py.File('/home/neda/Desktop/Neda_HMGU/from_luis/signal_10.mat','r')
variables_s = signal_file.items()

for var in variables_s:
    name_signal = var[0] # image_th
    data_signal = var[1]
    print "Name ", name_signal  
    if type(data_signal) is h5py.Dataset:
        # If DataSet pull the associated Data
        # If not a dataset, you may need to access the element sub-items
        signal = data_signal.value # NumPy Array / Value
 
#print value.shape # (4000, 180 , 50, 50)
# print signal.shape # (200, 180,4000)

print "loading images..." 
#image_file = h5py.File('C:\\Users\\Neda\\Desktop\\Neda\\from_luis\\image_10.mat','r') 
image_file = h5py.File('/home/neda/Desktop/Neda_HMGU/from_luis/image_10.mat','r')
variables_i = image_file.items()

for var in variables_i:    
    name_image = var[0] # image_th
    data_image = var[1]
    print "Name ", name_image  
    if type(data_image) is h5py.Dataset:
        # If DataSet pull the associated Data
        # If not a dataset, you may need to access the element sub-items
        image = data_image.value # NumPy Array / Value
 
#print value.shape # (50, 50 , 50, 50)
# print number of training-testing sample/ sizE(number of variable(signal vector))
#TODO normalize data histogram equalization .. (phD challnege) exclude zeros!



n_trees = 200
regr_multirf = RandomForestRegressor(n_estimators=n_trees,random_state=0)

fold = 0
error_list = []
#pred_y = []
n_folds = image.shape[0]
#cross validation
print "cross validation..."
#for train_idx, test_idx in k_fold.split(X):
for train_idx, test_idx in cross_validation.KFold(image.shape[0], n_folds = n_folds):
#for train_idx, test_idx in cross_validation.KFold(10, n_folds = 10):
    fold += 1

    #test_idx = np.asarray([train_idx[0]])

    signal_vec = signal.reshape(signal.shape[0], signal.shape[1] * signal.shape[2])

    #TODO extract releavant features
    
    # apply PCA to reduce dimension (not sure if it makes sense with RF! instead of feature extraction)
    
    #pca = decomposition.PCA(n_components=48)# n_components should be equal or less than number of samples!
    #pca.fit(signal_vec)
    #
    
    #PCA(copy=True, n_components=2, whiten=False)
    #signal_reduced = pca.transform(signal_vec)
    
    image_vec = image.reshape(image.shape[0], image.shape[1] * image.shape[2])

    #X_train_vec, X_test_vec = signal_reduced[train_idx,:], signal_reduced[test_idx,:]


    X_train_vec, X_test_vec = signal_vec[train_idx,:], signal_vec[test_idx,:]
    Y_train_vec, Y_test_vec = image_vec[train_idx,:], image_vec[test_idx,:]
    #Y_train, Y_test = image[train_idx,:,:], image[test_idx,:,:]

    
    '''
    X_train_vec = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test_vec = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    Y_train_vec = Y_train.reshape(Y_train.shape[0], Y_train.shape[1] * Y_train.shape[2])
    Y_test_vec = Y_test.reshape(Y_test.shape[0], Y_test.shape[1] * Y_test.shape[2])
    '''
    # reduce dimension
    # RF deals with large number of predictor

    e_list = []
    print "splitted train and test data"
    regr_multirf.fit(X_train_vec, Y_train_vec)
    # test loop
    print "testing fold #", fold, " / ", n_folds
    for j in range(Y_test_vec.shape[0]):

        y_multirf = regr_multirf.predict(X_test_vec[j].reshape(1,-1)) #Warning: Reshape your data either using X.reshape(1, -1) if it contains a single sample.
        #y_multirf = regr_multirf.predict(X_test_vec[j]) #Warning: Reshape your data either using X.reshape(1, -1) if it contains a single sample.
        #print y_multirf.shape
        #pred_y.append(y_multirf)
        #error = np.sqrt(mean_squared_error(center_test[j], pred_center))
        
    Y_test = np.asarray(Y_test_vec.reshape(image.shape[1],image.shape[2]))
    
    pred_y = np.asarray(y_multirf)
    #TODO reshape?!  
    pred_y = pred_y.reshape(image.shape[1], image.shape[2])
    
    
    #coeff_error = regr_multirf.score(X_test_vec, Y_test_vec)
    #print "coeff ", coeff_error    
    
    
    
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(10,10))

    ax1.imshow(Y_test)
    ax1.set_title('true')
    ax2.imshow(pred_y)
    ax2.set_title('predicted')
    #f.subplots_adjust(hspace=0)
    #f.tight_layout()
    #plt.xlim(0, 50) 
    #plt.ylim(0,50)
    #my_file = 'n_trees: .png'
    #plt.savefig('C:\\Users\\Neda\\Desktop\\Neda\\code\\results\\nn.png')
    #plt.savefig(os.path.join(my_path, 'n.png'))       
    #plt.imsave('C:\\Users\\Neda\\Desktop\\Neda\\code\\results\\nn.png', pred_y)
    f.savefig('./result4_from_fft/n_trees200_proj_180_sample_501/' + 'n_fold in n_samples' + str(fold) + 'in' + str(n_folds) + '.png')

    #TODO add colorbar    
    #f.colorbar(ax1)
    #######plt.show()
    
    
    #compare_images(pred_y, Y_test, "true vs predicted")
    #print sklearn.metrics.jaccard_similarity_score(y_multirf, Y_test_vec, normalize=True, sample_weight=None)
    #print sklearn.metrics.log_loss(Y_test_vec,y_multirf ) # multimodal not supported
    print "MAE: ", sklearn.metrics.mean_absolute_error(Y_test_vec, y_multirf)
    print "MSE: ", sklearn.metrics.mean_squared_error(Y_test_vec, y_multirf)
    #print sklearn.metrics.mean_squared_log_error(Y_test_vec, y_multirf)# when targhets have exponential growth
    #print sklearn.metrics.median_absolute_error(Y_test_vec, y_multirf) # multiple doesn't support median error
    print "r2 score ", sklearn.metrics.r2_score(Y_test_vec, y_multirf)
    
    #error_list.append(error)

    # TODO record time for each fold

print "done in ", time.time() - start_t, "seconds"





