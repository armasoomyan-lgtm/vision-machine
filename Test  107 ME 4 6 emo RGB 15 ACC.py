### BY GOD 

###############################  part 4 ######################################
#############################################################################
################### defining  model   ########################################
################### defin model
import os
#os.kill(os.getpid(), 9)
#os._exit(0)  # restart
#quit()   ### after run code crashed 
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Input, Flatten, Layer
from keras.layers import Dense, Conv2D , Conv3D, MaxPool3D , Flatten, Activation , MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import random
from random import shuffle
from google.colab.patches import cv2_imshow
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from models import disjoint_augment_image_pair
from loss_functions import intensity_loss, depth_consistency_loss3
from utils import load, save, DataLoader
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
###################################
from tensorflow.keras.layers import Conv2D
import tensorflow as tf
import tf_slim as slim
from tensorDLT import solve_DLT
from tf_spatial_transform import transform
from tensorflow.python.keras.layers import Conv2D
import tf_spatial_transform_local
import constant
import keras.backend as K

grid_w = constant.GRID_W
grid_h = constant.GRID_H

###################### mountinggggggggg gdrive  #########################
from google.colab import drive
drive.mount('/content/gdrive')
#################### import zipfile 

import zipfile
from zipfile import *
with zipfile.ZipFile("/content/gdrive/My Drive/Clips30+clipper.zip") as existing_zip:
    existing_zip.extractall("/content/")

# with zipfile.ZipFile("/content/gdrive/My Drive/Deceptive.zip") as existing_zip:
    # existing_zip.extractall("/content/")
# with zipfile.ZipFile("/content/gdrive/My Drive/Truthful.zip") as existing_zip:
    # existing_zip.extractall("/content/")


#################################################
max_val_MESH,min_val_MESH,max_val_H1,min_val_H1= (10,1,10,1)
################################################################################
pathgd="/content/gdrive/My Drive/"
print('eager mode is enabel??????????',tf.executing_eagerly())
###################################################################################
############################################################
# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,4,figsize=(30,7))

    # summarize history for accuracy Decept_Detect
    axs[0].plot(range(1,len(model_history.history['Decept_Detect_categorical_accuracy'])+1),model_history.history['Decept_Detect_categorical_accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_Decept_Detect_categorical_accuracy'])+1),model_history.history['val_Decept_Detect_categorical_accuracy'])
    axs[0].set_title('Model Decept_Detect_categorical_accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    #axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].set_xticks(np.arange(1,len(model_history.history['Decept_Detect_categorical_accuracy'])+1))
    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for Decept_Detect_loss
    axs[1].plot(range(1,len(model_history.history['Decept_Detect_loss'])+1),model_history.history['Decept_Detect_loss'])
    axs[1].plot(range(1,len(model_history.history['val_Decept_Detect_loss'])+1),model_history.history['val_Decept_Detect_loss'])
    axs[1].set_title('Model Decept_Detect_loss')
    axs[1].set_ylabel('Decept_Detect_loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['Decept_Detect_loss'])+1))
    axs[1].legend(['train', 'val'], loc='best') 

    # summarize history for M_EXP_loss
    axs[2].plot(range(1,len(model_history.history['M_EXP_loss'])+1),model_history.history['M_EXP_loss'])
    axs[2].plot(range(1,len(model_history.history['val_M_EXP_loss'])+1),model_history.history['val_M_EXP_loss'])
    axs[2].set_title('Model M_EXP_loss')
    axs[2].set_ylabel('M_EXP_loss')
    axs[2].set_xlabel('Epoch')
    axs[2].set_xticks(np.arange(1,len(model_history.history['M_EXP_loss'])+1))
    axs[2].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[3].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[3].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[3].set_title('Model Loss')
    axs[3].set_ylabel('Loss')
    axs[3].set_xlabel('Epoch')
    #axs[3].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[3].set_xticks(np.arange(1,len(model_history.history['loss'])+1))
    axs[3].legend(['train', 'val'], loc='best')
       
    fig.savefig('plot.png')
    plt.show()
############################

def H_estimator(train_inputs_aug, train_inputs, train_depth):
    return H_model(train_inputs_aug, train_inputs, train_depth)
    


#Covert global homo into mesh
def H2Mesh(H2, patch_size):
    batch_size = tf.shape(H2)[0]
    h = patch_size/grid_h
    w = patch_size/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh, 1], shape=[3], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(tf.expand_dims(p, 0),2))
    ori_pt = tf.concat(ori_pt, axis=2)
    ori_pt = tf.tile(ori_pt,[batch_size, 1, 1])
    tar_pt = tf.matmul(H2, ori_pt)
    
    #print('tar_pt , H2, ori_pt shape ==',tar_pt.shape, H2.shape,ori_pt.shape)
    ## = (4, 3, 81) (4, 3, 3) (4, 3, 81)
    x_s = tf.slice(tar_pt, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(tar_pt, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(tar_pt, [0, 2, 0], [-1, 1, -1])

    #print('x_s , y_s, z_s shape ==',x_s.shape, y_s.shape,z_s.shape)
    ## = (4, 1, 81) (4, 1, 81) (4, 1, 81)
    H2_local = tf.concat([x_s/z_s, y_s/z_s], axis=1)
    H2_local = tf.transpose(H2_local, perm=[0, 2, 1])
    H2_local = tf.reshape(H2_local, [batch_size, grid_h+1, grid_w+1, 2])
    print('H2_local shape ==',H2_local.shape)  ##  == (4, 9, 9, 2)  
    
    return H2_local


def H_model(train_inputs_aug, train_inputs, train_depth, patch_size=512.):

    train_inputs_aug = Input(shape=( 512, 512, 6),batch_size=3)  ## or (None,None, 3)
    train_inputs = Input(shape=( 512, 512, 6),batch_size=3)  ## or (None,None, 3)
    train_depth = Input(shape=( 512, 512, 1),batch_size=3)  ## or (None,None, 3)
    batch_size = train_inputs_aug.shape[0]
    
    ############################## build_model(train_inputs_aug) #, feature2_warp_gt, feature3_warp_gt
    input1 = train_inputs_aug[...,0:3]
    input2 = train_inputs_aug[...,3:6]
    #H1_motion, H2_motion, mesh_motion = build_model(train_inputs_aug) #, feature2_warp_gt, feature3_warp_gt
    _vgg_1 = _vgg(input1, input2)  
    [H1_motion, H2_motion, mesh_motion] = _vgg_1([input1, input2])
    
    print('H1_motion, H2_motion, mesh_motion ========',H1_motion.shape, H2_motion.shape, mesh_motion.shape)  
    ### ======== (1, 8, 1) (1, 8, 1) (1, 9, 9, 2)
    ############################################
    
    #scale transformation matrix
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    ################################################### solve global homo H1/H2
    #H1 = solve_DLT(H1_motion, patch_size)
    solve_DLT_Layer_512_1 = solve_DLT_Layer_512()
    H1 = solve_DLT_Layer_512_1(H1_motion)  
    print ('H1.shape ==',H1.shape)## H1.shape ==  (64, 3, 3)
    
    #H2 = solve_DLT(H1_motion+H2_motion, patch_size)
    solve_DLT_Layer_512_2 = solve_DLT_Layer_512()
    H2 = solve_DLT_Layer_512_1(H1_motion+H2_motion)  ## H1.shape ==  (64, 3, 3)
    print ('H2.shape ==',H2.shape)## H1.shape ==  (64, 3, 3)
    
    #H1_mat = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    MatMulLayer_10 = MatMulLayer()
    H1_mat = MatMulLayer_10(M_tile_inv,H1,M_tile)
    print(' H1_mat shape after matmul ===',H1_mat.shape ) ##(64, 3, 3)
    
    #H2_mat = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    MatMulLayer_20 = MatMulLayer()
    H2_mat = MatMulLayer_20(M_tile_inv,H2,M_tile)
    print(' H2_mat shape after matmul ===',H2_mat.shape ) ##(64, 3, 3)
    
    ################################################prepare for calculating loss###
    ## warp images using H1/H2
    image2_tensor = train_inputs[..., 3:6]
    
    #warp2_H1 = transform(image2_tensor, H1_mat)
    transformLayer_3 = transformLayer()
    warp2_H1 = transformLayer_3(image2_tensor,H1_mat)
    print(' warp2_H1 shape  ===',warp2_H1.shape) ##(None, 512, 512, 3) 
    
    #warp2_H2 = transform(image2_tensor, H2_mat)
    transformLayer_4 = transformLayer()
    warp2_H2 = transformLayer_4(image2_tensor,H2_mat)
    print(' warp2_H2 shape  ===',warp2_H2.shape) ##(None, 512, 512, 3) 
    
    ######################################## warp all-one images using H1/H2
    #one = tf.ones_like(image2_tensor, dtype=tf.float32)
    tf_ones_like_layer = tf_ones_like()
    one = tf_ones_like_layer(image2_tensor)
    
    #one_warp_H1 = transform(one, H1_mat)
    transformLayer_5 = transformLayer()
    one_warp_H1 = transformLayer_5(one,H1_mat)
    print(' one_warp_H1 shape  ===',one_warp_H1.shape) ##(None, 512, 512, 3)
    
    #one_warp_H2 = transform(one, H2_mat)
    transformLayer_6 = transformLayer()
    one_warp_H2 = transformLayer_6(one,H2_mat)
    print(' one_warp_H2 shape  ===',one_warp_H2.shape) ##(None, 512, 512, 3)
    
    ################################################### initialize the mesh using H2
    #ini_mesh = H2Mesh(H2, patch_size)
    H2_Mesh_layer_1 = H2_Mesh_layer_512()
    ini_mesh = H2_Mesh_layer_1(H2)
    print('ini_mesh shape  ===',ini_mesh.shape) #=== (4, 9, 9, 2)
    
    ############################################ calculate the final predicted mesh
    mesh = ini_mesh + mesh_motion
    #depth = tf.concat([train_depth, train_depth, train_depth], 3)
    depth = layers.Concatenate(axis=3)([train_depth, train_depth, train_depth])
    print('depth.shape==',depth.shape) ##(None, 512, 512, 3)
    ############################################# warp the image/all-one image/ depth map using mesh
    #warp2_H3, one_warp_H3, warp2_depth = tf_spatial_transform_local.transformer(image2_tensor, one, depth, mesh)
    tf_local_transformer_1 = tf_local_transformer()
    warp2_H3, one_warp_H3, warp2_depth = tf_local_transformer_1(image2_tensor, one, depth, mesh)
    
    #warp2_depth = tf.expand_dims(tf.reduce_mean(warp2_depth, 3),3)
    tf_expand_reduce_mean_1 = tf_expand_reduce_mean()
    warp2_depth = tf_expand_reduce_mean_1(warp2_depth)
    
    #warp2_depth = tf.clip_by_value(warp2_depth,  0, 1)
    tf_clip_by_value_1 = tf_clip_by_value()
    warp2_depth =tf_clip_by_value_1(warp2_depth)

    print('warp2_depth ========',(warp2_depth.shape))
    print('mesh ========',(mesh.shape))
    print('warp2_H1 ========',(warp2_H1.shape))
    print('warp2_H2 ========',(warp2_H2.shape))
    print('warp2_H3 ========',(warp2_H3.shape))
    print('one_warp_H1 ========',(one_warp_H1.shape))
    print('one_warp_H2 ========',(one_warp_H2.shape))
    print('one_warp_H3 ========',(one_warp_H3.shape))
    
    ### for training 
    #return warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3
    
    ### for testing
    #return H1_motion, H2_motion, mesh_motion,warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3
    return Model(inputs=[train_inputs_aug, train_inputs, train_depth], outputs=[H1_motion, H2_motion, mesh_motion,warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3])

def _conv_block(x,n, num_out_layers, kernel_sizes, strides):
    x = Input(shape=( None, None, n))
    #conv1 = slim.conv2d(inputs=x, num_outputs=num_out_layers[0], kernel_size=kernel_sizes[0], activation_fn=tf.nn.relu, scope='conv1')
    conv1 = keras.layers.Conv2D(filters=num_out_layers[0], kernel_size=kernel_sizes[0], activation='relu',padding="same", name='conv1')(x)  

    #conv2 = slim.conv2d(inputs=conv1, num_outputs=num_out_layers[1], kernel_size=kernel_sizes[1], activation_fn=tf.nn.relu, scope='conv2')
    conv2 = keras.layers.Conv2D(filters=num_out_layers[1], kernel_size=kernel_sizes[1], activation='relu',padding="same", name='conv2')(conv1) 
    return Model(inputs=x, outputs=conv2)


def feature_extractor(image_tf):  ## [None, 512, 512, 3]
    image_tf = Input(shape=( 512, 512, 3))  ## or (None,None, 3)
    feature = []
    #with tf.compat.v1.variable_scope('conv_block1'):
    #conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1))      ## [None, 512, 512, 64]
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    cnn_block = _conv_block(image_tf,3, ([64, 64]), (3, 3), (1, 1))      ## 3 to 64 [None, 512, 512, 64]
    conv1 = cnn_block(image_tf)      ## 3 to 64[None, 512, 512, 64]
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)##[None, 256, 256, 64]
    
    
      
    #with tf.compat.v1.variable_scope('conv_block2'):
    #conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))      #[None, 256, 256, 64]
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME') ## [None, 128, 128, 64]
    cnn_block = _conv_block(maxpool1,64, ([64, 64]), (3, 3), (1, 1))     # 64 to 64 [None, 256, 256, 64]
    conv2 = cnn_block(maxpool1)      #64 to 64 [None, 256, 256, 64]
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)  ## [None, 128, 128, 64]
      
    
    #with tf.compat.v1.variable_scope('conv_block3'):
    #conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))    #[None, 128, 128, 128]
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')## [None, 64, 64, 128]
    cnn_block = _conv_block(maxpool2,64, ([128, 128]), (3, 3), (1, 1))     #64 to 128 [None, 128, 128, 128]
    conv3 = cnn_block(maxpool2)      # 64 to 128 [None, 128, 128, 128]
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)  ## [None, 64, 64, 128]
       
 
    #with tf.compat.v1.variable_scope('conv_block4'):
    #conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))    ## [None, 64, 64, 128]
    cnn_block = _conv_block(maxpool3,128, ([128, 128]), (3, 3), (1, 1))     #128 to 128 [None, 64, 64, 128]
    conv4 = cnn_block(maxpool3)      #128 to 128 [None, 64, 64, 128]    
    
    #conv1_r64 = tf.image.resize_images(conv1, [64, 64], method=0)
    #conv1_r64 = tf.image.resize(conv1, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 64]
    #conv2_r64 = tf.image.resize(conv2, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 64]
    #conv3_r64 = tf.image.resize(conv3, [64, 64], method=tf.image.ResizeMethod.BILINEAR)## [None, 64, 64, 128]
    
    conv1_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv1) ## [None, 64, 64, 64]
    conv2_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv2) ## [None, 64, 64, 64]
    conv3_r64 = tf.keras.layers.Resizing(64,64,interpolation='bilinear')(conv3) ## [None, 64, 64, 128]]
    print('conv1,2,3_r64.shape ',conv1_r64.shape,conv2_r64.shape,conv3_r64.shape,conv4.shape) ##(None, 64, 64, 64) (None, 64, 64, 64) (None, 64, 64, 128) (None, None, None, 128)
    
    #feature.append(tf.concat([conv1, conv2, conv3, conv4], 3))  ### for test  and debug
    #feature.append(tf.concat([conv4, conv1_r64, conv2_r64, conv3_r64], 3))##[None, 64, 64, 384]=[[None, 64, 64, 128][None, 64, 64, 64][None, 64, 64, 64][None, 64, 64, 128]]
    conc0=keras.layers.concatenate([conv4, conv1_r64, conv2_r64, conv3_r64])
    print('conc0',conc0.shape)  ## conc0 (None, 64, 64, 384)
    feature.append(conc0) ##[None, 64, 64, 384]=[[None, 64, 64, 128][None, 64, 64, 64][None, 64, 64, 64][None, 64, 64, 128]]
    maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4)  ## [None, 32, 32, 128]
    
    #with tf.compat.v1.variable_scope('conv_block5'):
    #conv5 = _conv_block(maxpool4, ([256, 256]), (3, 3), (1, 1))    #32
    cnn_block = _conv_block(maxpool4,128, ([256, 256]), (3, 3), (1, 1))     #128 to 256 [None, 32, 32, 256]
    conv5 = cnn_block(maxpool4)      #128 to 256 [None, 32, 32, 256]
    
    #conv1_r32 = tf.image.resize(conv1, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv2_r32 = tf.image.resize(conv2, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv3_r32 = tf.image.resize(conv3, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    #conv4_r32 = tf.image.resize(conv4, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    
    conv1_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv1) ##[None, 512, 512, 64] 
    conv2_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv2) ##[None, 256, 256, 64]
    conv3_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv3) ##[None, 128, 128, 128]
    conv4_r32 = tf.keras.layers.Resizing(32, 32,interpolation='bilinear')(conv4) ##[None, 64, 64, 128]
    print('conv1,2,3,4_r32.shape ',conv1_r32.shape,conv2_r32.shape,conv3_r32.shape,conv4_r32,conv5.shape) ##
    
    #feature.append(tf.concat([conv5, conv1_r32, conv2_r32, conv3_r32, conv4_r32], 3))
    conc1=keras.layers.concatenate([conv5, conv1_r32, conv2_r32, conv3_r32, conv4_r32])
    print('conc1',conc1.shape)  ## conc1 
    feature.append(conc1) ##
    
    maxpool5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv5)  ## [None, 16, 16, 256]
    
    #with tf.compat.v1.variable_scope('conv_block6'):                                      
    #conv6 = _conv_block(maxpool5, ([256, 256]), (3, 3), (1, 1))    #16
    cnn_block = _conv_block(maxpool5,256, ([256, 256]), (3, 3), (1, 1))     #256 to 256[None, 16, 16, 256]
    conv6 = cnn_block(maxpool5)      #256 to 256[None, 16, 16, 256]  
    
    #conv1_r16 = tf.image.resize(conv1, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv2_r16 = tf.image.resize(conv2, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv3_r16 = tf.image.resize(conv3, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv4_r16 = tf.image.resize(conv4, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    #conv5_r16 = tf.image.resize(conv5, [16, 16], method=tf.image.ResizeMethod.BILINEAR)
    
    conv1_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv1) ##
    conv2_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv2) ##
    conv3_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv3) ##
    conv4_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv4) ##
    conv5_r16 = tf.keras.layers.Resizing(16, 16,interpolation='bilinear')(conv5) ##
    print('conv1,2,3,4,5_r16.shape ',conv1_r16.shape,conv2_r16.shape,conv3_r16.shape,conv4_r16,conv5_r16,conv6.shape) ##
    
    #feature.append(tf.concat([conv6, conv1_r16, conv2_r16, conv3_r16, conv4_r16, conv5_r16], 3))
    conc2=keras.layers.concatenate([conv6, conv1_r16, conv2_r16, conv3_r16, conv4_r16, conv5_r16])
    print('conc1',conc2.shape)  ## conc2 
    feature.append(conc2) ##
    
    import keras.backend as K
    #autoencoder.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    ff=keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=-1))
    f_1norm = keras.layers.LayerNormalization(axis=[-1])(feature[-1])
    f_2norm = keras.layers.LayerNormalization(axis=[-1])(feature[-2])
    f_3norm = keras.layers.LayerNormalization(axis=[-1])(feature[-3])
    
    #print('feature.shape-1 ===', tf.nn.l2_normalize(feature[-1]).shape)### (None, 16, 16, 896)feature is list[0--2] 
    print('feature.shape-1 ===',f_1norm.shape,ff)### (None, 16, 16, 896)  ##feature is list[0--2] 
    print('feature.shape-2 ===', f_2norm.shape)### (None, 32, 32, 640)
    print('feature.shape-3 ===', f_3norm.shape)### (None, 64, 64, 384)
    return feature,Model(inputs=image_tf, outputs=feature)


# contextual correlation layer
def CCL(c1, warp,s1,s2,s3):
    c1 = Input(shape=( s1, s2, s3),batch_size=3)  ## or (16, 16, 896)
    warp = Input(shape=( s1, s2, s3),batch_size=3)  ## or (16, 16, 896)
    
    print('c1.shape ===', c1.shape)  ## (None, 16, 16, 896) or (None, None, None, None)
    print('warp.shape ===', warp.shape)  ## (None, 16, 16, 896)
    
    #shape = warp.get_shape().as_list()
    shape = warp.shape
    print('shape  shape[0] shape[1] ===',shape,shape[0],shape[1])  ### [64, 16, 16, 896] 16
    kernel = 3
    stride = 1
    rate = 1
    if shape[1] == 16:
      rate = 1
      stride = 1
    elif shape[1] == 32:
      rate = 2
      stride = 1
    else:
      rate = 3
      stride = 1
    
    # extract patches as convolutional filters
    ##patches = tf.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    ## Extract patches from images and put them in the "depth" output dimension
    # Ideally, given an input image tensor of size 1x225x225x3 (where 1 is the batch size),
    # I want to be able to get Kx32x32x3 as output,
    # where K is the total number of patches 
    # and 32x32x3 is the dimension of each patch.
    # Is there something in tensorflow that already achieves this?
    
    #patches = tf.compat.v1.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    #patches = keras.ops.image.extract_patches(warp,(kernel,kernel), (stride,stride),(rate,rate),padding="same")
    patches = ExtractPatchesLayer(kernel_size=kernel, strides=stride, rates=rate, padding="same")(warp)
    print('patches===',patches.shape)  ###  patches=== ((64, 16, 16, 8064)  3*3*896=8064  
    
    #patches = tf.reshape(patches, [shape[0], -1, kernel, kernel, shape[3]])
    ## [None, -1, 3, 3, 896]
    patches = keras.layers.Reshape((-1, kernel, kernel, shape[3]))(patches)
    print('patches===',patches.shape)  ## ((64, 256, 3, 3, 896)
    
    #matching_filters = tf.transpose(patches, [0, 2, 3, 4, 1])
    matching_filters = TransposeLayer(perm=[0, 2, 3, 4, 1])(patches)  
    print('matching_filters.shape===',matching_filters.shape)  ##(None, 3, 3, 896, 256)
    
    # using convolution to match
    match_vol = []
      ## shape[0]== None
    for i in range(shape[0]):
      #expc1i0 = tf.expand_dims(c1[i], [0])
      expc1i0 = ExpandDimsLayer(axis=0)(c1[i])
      matchfili = matching_filters[i]
      #print('expc1i0.shape  ,matchfili.shape ===',expc1i0.shape,matchfili.shape)
      ## (1, 16, 16, 896) (3, 3, 896, 256)
      ## (1, 32, 32, 640) (3, 3, 640, 1024)
      ## (1, 64, 64, 384) (3, 3, 384, 4096)
      
      #single_match = tf.nn.atrous_conv2d(expc1i0, matchfili, rate=rate, padding='SAME')
      single_match = tf.keras.layers.Conv2D(
        filters=matchfili.shape[-1],  # Number of output channels (derived from matchfili)
        kernel_size=(matchfili.shape[0], matchfili.shape[1]), # Filter dimensions
        dilation_rate=rate,
        padding='same',  # Keras uses lowercase 'same'
        use_bias=False,  # Typically no bias when directly mimicking a convolution operation
        #weights=[matchfili] # If you want to initialize with the 'matchfili' weights
      )(expc1i0)
      
      #print('single_match.shape==',single_match.shape) ##single_match.shape== (1, 16, 16, 256)
      match_vol.append(single_match)
    
    print(len(match_vol))  ## 64
    #match_vol = tf.concat(match_vol, axis=0)
    match_vol = layers.Concatenate(axis=0)(match_vol)
    print('match_vol.shape==',match_vol.shape) ##(64, 16, 16, 256)
    channels = match_vol.shape[3]
    print("channels=====",match_vol.shape[3])  ##256
   
    
    # scale softmax
    softmax_scale = 10
    #match_vol = tf.nn.softmax(match_vol*softmax_scale,3)
    match_vol = layers.Softmax(axis=3)(match_vol*softmax_scale)
    print('match_vol.shape==',match_vol.shape) ##(64, 16, 16, 256)
    
    # convert the correlation volume to feature flow
    h_one = tf.linspace(0., shape[1]-1., int(match_vol.shape[1]))
    w_one = tf.linspace(0., shape[2]-1., int(match_vol.shape[2]))
    h_one = tf.matmul(tf.expand_dims(h_one, 1), tf.ones(shape=tf.stack([1, shape[2]])))
    w_one = tf.matmul(tf.ones(shape=tf.stack([shape[1], 1])), tf.transpose(tf.expand_dims(w_one, 1), [1, 0]))
    h_one = tf.tile(tf.expand_dims(tf.expand_dims(h_one, 0),3), [shape[0],1,1,channels])
    w_one = tf.tile(tf.expand_dims(tf.expand_dims(w_one, 0),3), [shape[0],1,1,channels])
    
    i_one = tf.expand_dims(tf.linspace(0., channels-1., channels),0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.tile(i_one, [shape[0], shape[1], shape[2], 1])
 
    flow_w = match_vol*(i_one%shape[2] - w_one)
    flow_h = match_vol*(i_one//shape[2] - h_one)
    
    #flow_w = tf.expand_dims(tf.reduce_sum(flow_w,3),3)
    # Reduce sum along axis 3 using a Lambda layer
    reduced_flow_w = layers.Lambda(lambda x: tf.reduce_sum(x, axis=3))(flow_w)
    #flow_w = tf.expand_dims(reduced_flow_w,3)
    flow_w = ExpandDimsLayer(axis=3)(reduced_flow_w)
    
    #flow_h = tf.expand_dims(tf.reduce_sum(flow_h,3),3)
    # Reduce sum along axis 3 using a Lambda layer
    reduced_flow_h = layers.Lambda(lambda x: tf.reduce_sum(x, axis=3))(flow_h)
    #flow_h = tf.expand_dims(reduced_flow_h,3)
    flow_h = ExpandDimsLayer(axis=3)(reduced_flow_h)
    
    
    #flow = tf.concat([flow_w, flow_h], 3)
    flow = layers.Concatenate(axis=3)([flow_w, flow_h])
    print("flow.shape",flow.shape)  ### (64, 16, 16, 2)  ,  (64, 32, 32, 2)  ,(64, 64, 64, 2)

    return Model(inputs=[c1,warp], outputs=flow)
    #return flow
    
def regression_H4pt_Net1(correlation):
    correlation = Input(shape=(16, 16, 2))  ## 64,16, 16, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  

    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  

    
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
   
    #fc1 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    
    fc1 = keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv3)  
    fc2 = keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=8, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)

    #H1_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    tf_expand_squeeze_layer = tf_expand_squeeze()
    H1_motion = tf_expand_squeeze_layer(fc3)
    
    
    return Model(inputs=correlation, outputs=H1_motion)
    
def regression_H4pt_Net2(correlation):
    correlation = Input(shape=(32, 32, 2))  ## 64,32, 32, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  
    
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
    
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    #conv4 = slim.conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv4 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool3)  
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv4')(conv4)  
   
    #fc1 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    fc1 = keras.layers.Conv2D(filters=128, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv4)  
    fc2 = keras.layers.Conv2D(filters=128, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=8, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)

    #H2_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    tf_expand_squeeze_layer = tf_expand_squeeze()
    H2_motion = tf_expand_squeeze_layer(fc3)
    
    return Model(inputs=correlation, outputs=H2_motion)
    
def regression_H4pt_Net3(correlation):
    correlation = Input(shape=(64, 64, 2))  ## 64, 64, 64, 2
    #conv1 = slim.conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv1 = slim.conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(correlation)  
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv1')(conv1)  
    
    #maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    #conv2 = slim.conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    #conv2 = slim.conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1)
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same")(maxpool1)  
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu',padding="same", name='conv2')(conv2)  
    
    #maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    #conv3 = slim.conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv3 = slim.conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2)
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool2)  
    conv3 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv3')(conv3)  
    
    #maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    #conv4 = slim.conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    #conv4 = slim.conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3)
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same")(maxpool3)  
    conv4 = keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',padding="same", name='conv4')(conv4)  
    
    #maxpool4 = slim.max_pool2d(conv4, 2, stride=2, padding = 'SAME')
    #conv5 = slim.conv2d(inputs=maxpool4, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    #conv5 = slim.conv2d(inputs=conv5, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    maxpool4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding="same")(maxpool4)  
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu',padding="same", name='conv5')(conv5)  
  
    #fc1 = slim.conv2d(inputs=conv5, num_outputs=2048, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    #fc2 = slim.conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    #fc3 = slim.conv2d(inputs=fc2, num_outputs=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation_fn=None)

    fc1 = keras.layers.Conv2D(filters=2048, kernel_size=4, activation='relu',padding="valid", name='fc1')(conv5)  
    fc2 = keras.layers.Conv2D(filters=1024, kernel_size=1, activation='relu',padding="same", name='fc2')(fc1)
    fc3 = keras.layers.Conv2D(filters=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation=None,padding="same", name='fc3')(fc2)


    print('fc3 shape ===',fc3.shape)
    #net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    #mesh_motion = tf.reshape(fc3, (-1, grid_h+1, grid_w+1, 2))
    mesh_motion = keras.layers.Reshape((grid_h+1, grid_w+1, 2))(fc3)
    print('mesh_motion shape ===',mesh_motion.shape)
    
    return Model(inputs=correlation, outputs=mesh_motion)



def _vgg(input1, input2):
    input1 = Input(shape=( 512, 512, 3),batch_size=3)  ## or (None,None, 3)
    input2 = Input(shape=( 512, 512, 3),batch_size=3)  ## or (None,None, 3)
    batch_size = input1.shape[0]
    print('batch_size ===',batch_size)
    
    # ## feature extractors with shared weights
    # with tf.compat.v1.variable_scope('feature_extract', reuse = None): 
      # feature1 = feature_extractor(input1)
    # with tf.compat.v1.variable_scope('feature_extract', reuse = True): 
      # feature2 = feature_extractor(input2)
     
    f,feature_extract_ = feature_extractor(input1)    
    feature1 = feature_extract_(input1)
    f,feature_extract_ = feature_extractor(input2)    
    feature2 = feature_extract_(input2)
     
    print('feature1[-1] feature2[-1] ==' ,feature1[-1].shape, feature2[-1].shape) 
    ##(64, 16, 16, 896) (64, 16, 16, 896)
     
    ################################################# the 1st layer of the pyramid
    #featureflow_1 = CCL(tf.nn.l2_normalize(feature1[-1],axis=3), tf.nn.l2_normalize(feature2[-1],axis=3))
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-1]) ##(64, 16, 16, 896)
    b = keras.layers.LayerNormalization(axis=[-1])(feature2[-1]) ##(64, 16, 16, 896)
    CCL_1 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_1 = CCL_1([a,b]) ## (64, 16, 16, 2)
    #CCL_Layar_1 = CCL_Layar()
    #featureflow_1 = CCL_Layar_1(a,b)
 
    regression_H4pt_Net1_1 = regression_H4pt_Net1(featureflow_1)
    H1_motion = regression_H4pt_Net1_1(featureflow_1)
    print('H1_motion shape ===',H1_motion.shape)
    ##(None, 8,1)
    
    # warp the feature map
    patch_size = 32.
    #H1 = solve_DLT(H1_motion/16., patch_size)
    solve_DLT_Layer_1 = solve_DLT_Layer_32()
    H1 = solve_DLT_Layer_1(H1_motion/16.)  
    ## H1.shape ==  (64, 3, 3)
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    print('M_tile_inv , H1  M_tile shape ===',M_tile_inv.shape,H1.shape,M_tile.shape )##(64, 3, 3) (64, 3, 3)(64, 3, 3)
    #H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    MatMulLayer_1 = MatMulLayer()
    H1 = MatMulLayer_1(M_tile_inv,H1,M_tile)
    print(' H1 shape after matmul ===',H1.shape ) ##(64, 3, 3)
    
    #feature2_warp = transform(tf.nn.l2_normalize(feature2[-2],axis=3), H1)
    a = keras.layers.LayerNormalization(axis=[-1])(feature2[-2])
    transformLayer_1 = transformLayer()
    feature2_warp = transformLayer_1(a,H1)
    print(' feature2_warp shape  ===',feature2_warp.shape,a.shape,H1.shape ) ##(None, 32, 32, 640) (None, 32, 32, 640) (64, 3, 3)
    
    ############################################################### the 2nd layer of the pyramid
    ##(64, 32, 32, 640) (64, 32, 32, 640)
    #featureflow_2 = CCL(tf.nn.l2_normalize(feature1[-2],axis=3), feature2_warp) ## (64, 32, 32, 2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-2])
    b = feature2_warp  ##(64, 32, 32, 640)
    CCL_2 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_2 = CCL_2([a,b])
    print("featureflow_2.shape",featureflow_2.shape)  ##(64, 32, 32, 2)
    
    
    #H2_motion = regression_H4pt_Net2(featureflow_2)
    regression_H4pt_Net2_1 = regression_H4pt_Net2(featureflow_2)
    H2_motion = regression_H4pt_Net2_1(featureflow_2)
    print('H2_motion shape ===',H2_motion.shape) ##(64, 8, 1)
   
    
    # warp the feature map
    patch_size = 64.
    #H2 = solve_DLT((H1_motion+H2_motion)/8., patch_size)
    solve_DLT_Layer_2 = solve_DLT_Layer_64()
    H2 = solve_DLT_Layer_2((H1_motion+H2_motion)/8.)  
    print('H2 shape ===',H2.shape)  ## H2.shape ==  (64, 3, 3)
    
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    #H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    MatMulLayer_2 = MatMulLayer()
    H2 = MatMulLayer_2(M_tile_inv,H2,M_tile)
    print(' H2 shape after matmul ===',H2.shape ) ##(64, 3, 3)
    
    #feature3_warp = transform(tf.nn.l2_normalize(feature2[-3],axis=3), H2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature2[-3])
    transformLayer_2 = transformLayer()
    feature3_warp = transformLayer_2(a,H2)
    print(' feature3_warp shape  ===',feature3_warp.shape,a.shape,H2.shape ) ##(None, 64, 64, 384) (None, 64, 64, 384) (64, 3, 3)
    
    
    ################################################################# the 3rd layer of the pyramid
    ## (64, 64, 64, 384) (64, 64, 64, 384)
    #featureflow_3 = CCL(tf.nn.l2_normalize(feature1[-3],axis=3), feature3_warp) ##(64, 64, 64, 2)
    a = keras.layers.LayerNormalization(axis=[-1])(feature1[-3])
    b = feature3_warp  ##(64, 32, 32, 640)
    CCL_3 = CCL(a,b,a.shape[1],a.shape[2],a.shape[3])  
    featureflow_3 = CCL_3([a,b])
    print("featureflow_3.shape",featureflow_3.shape)  ##(64, 64, 64, 2)
    
    #mesh_motion = regression_H4pt_Net3(featureflow_3)
    regression_H4pt_Net3_1 = regression_H4pt_Net3(featureflow_3)
    mesh_motion = regression_H4pt_Net3_1(featureflow_3)
    print('mesh_motion shape ===',mesh_motion.shape)  ##(None, 9, 9, 2)
    
    
    return Model(inputs=[input1, input2], outputs=[H1_motion, H2_motion, mesh_motion])
    #return H1_motion, H2_motion, mesh_motion
#######################################################################################
class AugmentLayer(Layer):
    def call(self, inputs):
        return disjoint_augment_image_pair(inputs)  # This line should be indented
    def compute_output_shape(self, input_shape):
        # Assuming disjoint_augment_image_pair preserves the shape
        return input_shape

class H2_Mesh_layer_512(tf.keras.layers.Layer):
    def call(self, H2):
        return H2Mesh(H2, 512.)
        
class tf_local_transformer(tf.keras.layers.Layer):
    def call(self, image2_tensor, one, depth, mesh):
        return tf_spatial_transform_local.transformer(image2_tensor, one, depth, mesh)
        

class AbsoluteDifferenceLayer(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return K.abs(x - y)
        
class tf_ones_like(tf.keras.layers.Layer):
    def call(self, image2_tensor):
        return tf.ones_like(image2_tensor, dtype=tf.float32)
        
class Hmodel_512(tf.keras.layers.Layer):
    def call(self, train_inputs_aug, train_inputs, train_depth):
                ##Ensure all inputs are tensors
        train_inputs_aug = tf.convert_to_tensor(train_inputs_aug)
        train_inputs = tf.convert_to_tensor(train_inputs)
        train_depth = tf.convert_to_tensor(train_depth)
        return H_model(train_inputs_aug, train_inputs, train_depth,512)

        
class solve_DLT_Layer_32(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,32.)
        
class solve_DLT_Layer_64(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,64.)
        
class solve_DLT_Layer_512(tf.keras.layers.Layer):
    def call(self, x):
        #y = np.float32(y)
        #y = tf.cast(y, tf.float32)
        #y = tf.float32(y)
        return solve_DLT(x,512.)
        
class MatMulLayer(tf.keras.layers.Layer):
    def call(self, A, B, C):
        return tf.matmul(tf.matmul(A, B), C)
        
class transformLayer(tf.keras.layers.Layer):
    def call(self, A, B):
        return transform(A, B)
   
 
        
class tf_expand_squeeze(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(tf.squeeze(tf.squeeze(x,1),1), [2])

class tf_expand_reduce_mean(tf.keras.layers.Layer):
    def call(self, warp2_depth):
        return tf.expand_dims(tf.reduce_mean(warp2_depth, 3),3)

class tf_clip_by_value(tf.keras.layers.Layer):
    def call(self, warp2_depth):
        return tf.clip_by_value(warp2_depth,  0, 1) 
     
class CCL_Layar(tf.keras.layers.Layer):
    def call(self,c1, warp):
        return CCL(c1, warp)
       
class ExtractPatchesLayer(keras.layers.Layer):
    def __init__(self, kernel_size, strides, rates, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = self._validate_tuple(kernel_size, "kernel_size")
        self.strides = self._validate_tuple(strides, "strides")
        self.rates = self._validate_tuple(rates, "rates")
        self.padding = padding.upper()  # Keras ops expects uppercase padding

    def _validate_tuple(self, value, name):
        if isinstance(value, int):
            return (value, value)
        elif isinstance(value, tuple) and len(value) == 2:
            return value
        else:
            raise ValueError(
                f"'{name}' must be an integer or a tuple of two integers. "
                f"Received: {value}"
            )

    def call(self, inputs):
        return keras.ops.image.extract_patches(
            inputs,
            size =self.kernel_size,
            strides=self.strides,
            dilation_rate=self.rates,
            padding=self.padding
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "rates": self.rates,
            "padding": self.padding,
        })
        return config
class ExpandDimsLayer(keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config

class TransposeLayer(keras.layers.Layer):
    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)

    def get_config(self):
        config = super().get_config()
        config.update({'perm': self.perm})
        return config


############# Define a function to calculate the loss
def compute_loss(train_inputs_aug, train_inputs, train_depth):
    # content loss
    #with tf.compat.v1.variable_scope('generator', reuse=tf.compat.v1.AUTO_REUSE):  # Added reuse=tf.compat.v1.AUTO_REUSE
    #H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = H_estimator(train_inputs_aug, train_inputs, train_depth)
    
    # ... In compute_loss function ...
    h_estimator_layer = HEstimatorLayer()  # Create an instance
    H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2, train_warp2_H3, train_one_warp_H1, train_one_warp_H2, train_one_warp_H3 = h_estimator_layer(train_inputs_aug, train_inputs, train_depth) 


    lam_lp = 1
    loss1 = intensity_loss(gen_frames=train_warp2_H1, gt_frames=train_inputs[...,0:3]*train_one_warp_H1, l_num=1)
    loss2 = intensity_loss(gen_frames=train_warp2_H2, gt_frames=train_inputs[...,0:3]*train_one_warp_H2, l_num=1)
    loss3 = intensity_loss(gen_frames=train_warp2_H3, gt_frames=train_inputs[...,0:3]*train_one_warp_H3, l_num=1)
    lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
    # mesh loss
    lam_mesh = 0   ##   deph assistance??   without lam_mesh = 0      within 10
    mesh_loss = depth_consistency_loss3(train_warp2_depth, train_mesh)
    # total loss
    g_loss = tf.add_n([lp_loss * lam_lp, mesh_loss * lam_mesh], name='g_loss')
    
    print('mesh_loss==',tf.shape(mesh_loss),format(mesh_loss),mesh_loss)
    print('lp_loss==',tf.shape(lp_loss),format(lp_loss),lp_loss)
    print('g_loss==',tf.shape(g_loss),format(g_loss),g_loss)
    print('lam_mesh==',tf.shape(lam_mesh),format(lam_mesh),lam_mesh)
     
    return g_loss, lp_loss, mesh_loss # Return all losses
##################################################################
###################################################################
# homograph_pic = np.load('/content/gdrive/My Drive/homograph_pic_np.npy')
# #### homograph_pic.shape (8, 3, 512, 512, 6) <class 'tuple'> 5
# ## normalize_image(frame2 / 127.5) - 1.0

# batch_size = 3   ###constant.TEST_BATCH_SIZE
# height, width = 512, 512
# ##test_inputs_clips_tensor =tf.ones(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
# ##test_inputs_clips_tensor =tf.random.normal(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
# ##test_inputs_clips_np=np.random.rand(batch_size, height, width, 3 * 2).astype(np.float32)
# ##test_inputs = test_inputs_clips_np
# test_inputs = homograph_pic[129,:,:,:,:]
# print('test inputs = {}'.format(test_inputs))
# ##########  depth is not needed in the inference process, 
# ##we assign "test_depth" arbitrary values such as an all-one map
# ##test_depth = tf.ones_like(test_inputs[...,0:1])
# ##test_depth = test_inputs[...,0:1]
# ##test_depth =  tf.random.normal(shape=tf.shape(test_inputs[...,0:1]), dtype=test_inputs.dtype)
# test_depth=np.random.rand(batch_size, height, width, 1).astype(np.float32)
# print("test_depth.shape")
# train_inputs_aug = disjoint_augment_image_pair(test_inputs)
# g_loss, lp_loss, mesh_loss = compute_loss(train_inputs_aug, test_inputs, test_depth)

# cv2_imshow(cv2.resize((test_inputs[0][:,:,3:6]+1)*127.5,(64,64),interpolation = cv2.INTER_CUBIC))
# ### Convert train_inputs_aug to a NumPy array before using cv2_imshow
# train_inputs_aug_np = train_inputs_aug.numpy()  # Convert to NumPy array
# cv2_imshow(cv2.resize((train_inputs_aug_np[0][:,:,3:6]+1)*127.5,(64,64),interpolation = cv2.INTER_CUBIC))

#######################################################
##########################################

input_1_0 =  keras.layers.Input(shape=(15, 64, 64, 3))  ## input should be Alone  and uint8

def normalize2(x):
    return (tf.cast(x, tf.float32) / 255.0) - 0.0
input_1 = keras.layers.Lambda(normalize2)(input_1_0)

#output_1 = keras.layers.BatchNormalization()(input_1) ## ----
# 1111111 output_1.shape  (None, 15, 64, 64, 64)
# 1111111 output_1.shape  (None, 7, 32, 32, 64)
# 222222 output_1.shape  (None, 7, 32, 32, 128)
# 222222 output_1.shape  (None, 3, 16, 16, 128)
# 333333 output_1.shape  (None, 3, 16, 16, 256)
# 3333333 output_1.shape  (None, 3, 8, 8, 256)
# 4444444 output_1.shape  (None, 3, 8, 8, 512)
# 4444444 output_1.shape  (None, 3, 4, 4, 512)
# 555555 output_1.shape  (None, 107)
# 666666 output_2.shape  (None, 105)
# 777777 output_3.shape  (None, 48)
# 888888 output_4.shape  (None, 972)
# ok ok input_5.shape  (None, 512, 512, 6)
# 99999 outcombine.shape  (None, 256)
# 10-10-10-10 outcombine.shape  (None, 1024)
# 11-11-11-11 Decept_Detect.shape  (None, 2)


output_1 = input_1

#output_1 = keras.layers.Conv3D(64, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(64, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('1111111 output_1.shape ',output_1.shape) ## (None, 15, 64, 64, 64)
output_1 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_1)
print('1111111 output_1.shape ',output_1.shape) ##(None, 7, 32, 32, 64)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(128, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(128, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('222222 output_1.shape ',output_1.shape) ## (None, 7, 32, 32, 128)
output_1 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_1)
print('222222 output_1.shape ',output_1.shape) ## (None, 3, 16, 16, 128)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(256, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(256, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('333333 output_1.shape ',output_1.shape) ## (None, 3, 16, 16, 256)
output_1 = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(output_1)
print('3333333 output_1.shape ',output_1.shape) ## (None, 3, 8, 8, 256)
output_1 = keras.layers.Dropout(0.3)(output_1)

#output_1 = keras.layers.Conv3D(512, (5, 3, 3), activation='relu', padding='same')(output_1)
output_1 = keras.layers.Conv3D(512, (5, 3, 3), padding='same')(output_1)
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = keras.activations.relu(output_1)  ## ++++
print('4444444 output_1.shape ',output_1.shape) ## (None, 3, 8, 8, 512)
output_1 = keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(output_1)
print('4444444 output_1.shape ',output_1.shape) ## (None, 3, 4, 4, 512
output_1 = keras.layers.Dropout(0.3)(output_1)


output_1 = keras.layers.Flatten()(output_1)

#output_1 = keras.layers.Dense(1024,activation='relu',name='M_EXP')(output_1) ## fc1
output_1 = keras.layers.Dense(1024)(output_1) ## fc1
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
output_1 = layers.Activation('relu')(output_1)
output_11 = output_1

#output_1 = keras.layers.Dense(107,activation='softmax',name='M_EXP')(output_1) ## fc2
output_1 = keras.layers.Dense(107)(output_1) ## fc2
output_1 = keras.layers.BatchNormalization()(output_1)  ## ++++
#output_1 = keras.activations.softmax(name='M_EXP')(output_1)  ## ++++
output_1 = layers.Activation('softmax',name='M_EXP')(output_1)
M_EXP = output_1  #### branching from output_1
print('555555 output_1.shape ',output_1.shape) ##  (None, 107)
# output_1 = keras.layers.Flatten()(output_1)
#output_1 = keras.layers.BatchNormalization()(output_1)   ### ?????? Can delete


input_2 =  keras.layers.Input(shape=(15,1,7))
#output_2 = keras.layers.Conv2D(32,(5,5), activation='relu')(input_2)
#output_2 = keras.layers.MaxPooling2D(pool_size=(2,2))(output_2)
output_2 = keras.layers.Flatten()(input_2)
print('666666 output_2.shape ',output_2.shape) ##
##output_2 = keras.layers.BatchNormalization()(output_2)    ###### ?????? Can delete -----


input_3 =  keras.layers.Input(shape=(6,8,1))
#output_3 = keras.layers.Conv2D(32,(5,5), activation='relu')(input_3)
#output_3 = keras.layers.MaxPooling2D(pool_size=(2,2))(output_3)
#output_3 = keras.activations.sigmoid(output_3)   ## OUTPUT RANGE === 0 to 1
#output_3 = keras.layers.BatchNormalization()(output_3)     ###(output_3)
output_3 = keras.layers.Flatten()(input_3)
print('777777 output_3.shape ',output_3.shape) ##
#output_3 = keras.layers.BatchNormalization()(output_3)     ###-----

input_4 =  keras.layers.Input(shape=(6,9,9,2))
#output_4 = keras.layers.Conv3D(64, (5, 3, 3), activation='relu', padding='same')(input_4)
#output_4 = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(output_4)
#output_4 = keras.activations.sigmoid(output_4)   ## OUTPUT RANGE === 0 to 1
#output_4 = keras.layers.BatchNormalization()(output_4)              ###(output_4)
output_4 = keras.layers.Flatten()(input_4)
print('888888 output_4.shape ',output_4.shape) ##
#output_4 = keras.layers.BatchNormalization()(output_4)##-------

##//////////////////##????????????????????????????###////////////////////////    
################
# input_5_0 = input_1_0  ## b,15,64, 64, 3 //input should be Alone  and uint8

# # frame1_tensor = input_5_0[:][2+j*4][:,:,:] ## select frame 2,6,10,14 from 20=pack  (b ,64,64,3)
# # frame1_tensor = input_5_0[:][2+4+j*4][:,:,:] ## select frame 6,10,14,18 from 20=pack (b ,64,64,3)

# frame1_tensor = input_5_0[:,2,:,:,:] ## select frame 2, from 20=pack  (b ,64,64,3)
# frame2_tensor = input_5_0[:,2+4,:,:,:] ## select frame 6, from 20=pack (b ,64,64,3)

# # Normalization (can be done with a Lambda layer for clarity)
# def normalize1(x):
    # return (tf.cast(x, tf.float32) / 127.5) - 1.0
# frame1_normalized = keras.layers.Lambda(normalize1)(frame1_tensor)
# frame2_normalized = keras.layers.Lambda(normalize1)(frame2_tensor)
# # Resizing 
# frame1_resized = tf.keras.layers.Resizing(512,512,interpolation='bicubic')(frame1_normalized)
# frame2_resized = tf.keras.layers.Resizing(512,512,interpolation='bicubic')(frame2_normalized)
# # Concatenate the processed frames
# input_5 = keras.layers.Concatenate(axis=3)([frame1_resized, frame2_resized]) ## (b=3,512,512,6)
# #########################

# print('ok ok input_5.shape ',input_5.shape)  ## b=3,512, 512, 6
# batch_size = input_5.shape[0]
# print('batch_size ========',batch_size)

# #test_depth=np.random.rand( 512, 512, 1).astype(np.float32)   ### 3,512, 512, 6
# #train_inputs_aug = disjoint_augment_image_pair(input_5)

# augment_layer = AugmentLayer()
# train_inputs_aug = augment_layer(input_5)

# H_model_1 = H_model(train_inputs_aug, input_5, input_5[...,0:1],patch_size=512.)  
# [H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3, train_one_warp_H1, train_one_warp_H2,train_one_warp_H3] = H_model_1([train_inputs_aug, input_5, input_5[...,0:1]])
# #H1_motion, H2_motion, mesh_motion,train_warp2_depth, train_mesh, train_warp2_H1, train_warp2_H2,train_warp2_H3, train_one_warp_H1, train_one_warp_H2,train_one_warp_H3 = H_model(train_inputs_aug, input_5, input_5[...,0:1], patch_size=512.)

# l_num = 1.
# lam_lp = 1.

# abs_diff_layer = AbsoluteDifferenceLayer()

# # [gen_frames=train_warp2_H1, gt_frames=input_5[...,0:3]*train_one_warp_H1])
# # [gen_frames=train_warp2_H2, gt_frames=input_5[...,0:3]*train_one_warp_H2])
# # [gen_frames=train_warp2_H3, gt_frames=input_5[...,0:3]*train_one_warp_H3])

# # loss1 = abs_diff_layer[(train_warp2_H1, input_5[...,0:3]*train_one_warp_H1)]
# # loss2 = abs_diff_layer[(train_warp2_H2, input_5[...,0:3]*train_one_warp_H2)]
# # loss3 = abs_diff_layer[(train_warp2_H3, input_5[...,0:3]*train_one_warp_H3]
# # lp_loss = 16. * loss1 + 4. * loss2 + 1. * loss3
# # g_loss = lp_loss * lam_lp 

# def loss_operation(train_warp2_H1, input_5, train_one_warp_H1,
                     # train_warp2_H2, train_one_warp_H2,
                     # train_warp2_H3, train_one_warp_H3):

    # loss1 = train_warp2_H1 - input_5[..., 0:3] * train_one_warp_H1
    # loss2 = train_warp2_H2 - input_5[..., 0:3] * train_one_warp_H2
    # loss3 = train_warp2_H3 - input_5[..., 0:3] * train_one_warp_H3

    # lp_loss = 16.0 * (loss1*loss1) + 4.0 * (loss2*loss2) + 1.0 * (loss3*loss3)
    # lam_lp = 1.
    # g_loss000 = lp_loss * lam_lp #(3=b, 512, 512, 3)
    # # Sum along axes 1, 2, and 3 using tf.reduce_sum()
    # g_loss00 = tf.reduce_sum(g_loss000, axis=(1, 2, 3)) ##Shape'g_loss00' (after summing axes 1, 2, 3): (b=3,)

    # return g_loss00

# # Create a Lambda layer to perform the custom operation
# tt_layer = layers.Lambda(lambda tensors: loss_operation(tensors[0], tensors[1], tensors[2],
                                                           # tensors[3], tensors[4], tensors[5],
                                                           # tensors[6]),
                                                 # )(
    # [train_warp2_H1, input_5, train_one_warp_H1,
     # train_warp2_H2, train_one_warp_H2,
     # train_warp2_H3, train_one_warp_H3]
# )
# g_loss = tt_layer

# # loss1 = train_warp2_H1 - input_5[...,0:3]*train_one_warp_H1
# # loss2 = train_warp2_H2 - input_5[...,0:3]*train_one_warp_H2
# # loss3 = train_warp2_H3 - input_5[...,0:3]*train_one_warp_H3
# # lp_loss = 16. * loss1* loss1 + 4. * loss2* loss2 + 1. * loss3* loss3
# # g_loss = lp_loss * lam_lp 

# print('g_loss.shape======',g_loss.shape,g_loss)  ##(3=b,) , dtype=float32, sparse=True, name=keras_tensor_604>
# #g_loss = keras.layers.Flatten()(g_loss)  ## xx=None*512*512*3 --- (3=b, 786432)
# ####### ????? g_loss = keras.activations.sigmoid(g_loss)   ## OUTPUT RANGE === 0 to 1 !!!!!!!!! use tanh 
# g_loss = keras.layers.Flatten(name='g_loss')(g_loss)  ##  only fo naming name='g_loss' for plot acc , loss xx=None*512*512*3 --- (3, 786432)
# print('g_loss.shape======',g_loss.shape,g_loss)   ### (3=b,)
# #g_loss = layers.Embedding( sparse=True)(g_loss)
# out01 = g_loss
        
# output_5 = keras.layers.Flatten()(mesh_motion)
# mesh = keras.layers.Dense(1,activation='sigmoid',name='mesh')(output_5)
# print('mesh.shape ========',mesh) #<KerasTensor shape=(3, 1), dtype=float32, sparse=False, name=keras_tensor_606>



#out00 = np.random.rand(1).astype(np.float32)
#output_5 = keras.layers.Flatten()(input_5)
#out00 = keras.layers.Dense(1,activation='softmax',name='out00')(output_5)
##//////////////////##????????????????????????????###////////////////////////

#out12345 = [output_1,output_2,output_3,output_4,output_5]
out12345 = [output_11,M_EXP,output_2,output_3,output_4]
outcombine = keras.layers.concatenate(out12345)

#outcombine = keras.layers.Dense(256,activation='relu')(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(256)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
outcombine = keras.activations.relu(outcombine)  ## ++++
print('99999 outcombine.shape ',outcombine.shape) ##



#outcombine = keras.layers.Dense(1024,activation='relu')(outcombine)  ## 256   --->  1024 Hidden_Layer
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(1024)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
outcombine = keras.activations.relu(outcombine)  ## ++++
print('10-10-10-10 outcombine.shape ',outcombine.shape) ##


#Decept_Detect = keras.layers.Dense(2,activation='softmax',name='Decept_Detect')(outcombine)## fc2
outcombine = keras.layers.Dropout(0.2)(outcombine)
outcombine = keras.layers.Dense(2)(outcombine)  ## 815   ---> 256 fc1
outcombine = keras.layers.BatchNormalization()(outcombine)  ## ++++
#Decept_Detect = keras.activations.softmax(name='Decept_Detect')(outcombine)  ## ++++
Decept_Detect = layers.Activation('softmax',name='Decept_Detect')(outcombine)
print('11-11-11-11 Decept_Detect.shape ',Decept_Detect.shape) ##

  
#MHEmodel = keras.models.Model([input_1_0,input_2,input_3,input_4],[M_EXP,Decept_Detect,mesh,g_loss])
MHEmodel = keras.models.Model([input_1_0,input_2,input_3,input_4],[M_EXP,Decept_Detect])


keras.utils.plot_model(MHEmodel,
    to_file="model1.jpg", ##to_file="model.png"
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True)
MHEmodel.summary()
print('len(MHEmodel.layers)===',len(MHEmodel.layers))    

#############################################################################
#############################################################################
###############################  part 5 ######################################
#############################################################################
#################################### load inputs && outputs  ###############
#all_video_np=np.random.randn(124,20, 112, 112, 3).astype(np.float32)
#### Train Data
all_video_np=np.load('/content/gdrive/My Drive/train_30clip_video_np.npy')
all_video_label_01_np=np.load('/content/gdrive/My Drive/train_30clip_label_01_np.npy')
micro_exp_labels=np.load('/content/gdrive/My Drive/train_30clip_micro_exp_labels.npy')
Emotion_label_np=np.load('/content/gdrive/My Drive/train_6464_30clip_Emotion_label_np.npy')
homograph_label_H1_np=np.load('/content/gdrive/My Drive/train_30clip_homograph_label_H1_np.npy')
homograph_label_MESH_np=np.load('/content/gdrive/My Drive/train_30clip_homograph_label_MESH_np.npy')

#homograph_pic = np.load('/content/gdrive/My Drive/homograph_pic_np.npy')
#### homograph_pic.shape (None, 3, 512, 512, 6) <class 'tuple'> 5
## normalize_image(frame2 / 127.5) - 1.0

#### Val  Data  ---- or  test  Data
val_vid_np =np.load('/content/gdrive/My Drive/val_30clip_video_np.npy')
val_vid_label_01_np=np.load('/content/gdrive/My Drive/val_30clip_label_01_np.npy')
val_mic_exp_labels=np.load('/content/gdrive/My Drive/val_30clip_micro_exp_labels.npy')
val_Emo_label_np=np.load('/content/gdrive/My Drive/val_6464_30clip_Emotion_label_np.npy')
val_homo_label_H1_np=np.load('/content/gdrive/My Drive/val_30clip_homograph_label_H1_np.npy')
val_homo_label_MESH_np=np.load('/content/gdrive/My Drive/val_30clip_homograph_label_MESH_np.npy')

#### test  Data  ---- or  test  Data
test_vid_np =np.load('/content/gdrive/My Drive/test_30clip_video_np.npy')
test_vid_label_01_np=np.load('/content/gdrive/My Drive/test_30clip_label_01_np.npy')
test_mic_exp_labels=np.load('/content/gdrive/My Drive/test_30clip_micro_exp_labels.npy')
test_Emo_label_np=np.load('/content/gdrive/My Drive/test_6464_30clip_Emotion_label_np.npy')
test_homo_label_H1_np=np.load('/content/gdrive/My Drive/test_30clip_homograph_label_H1_np.npy')
test_homo_label_MESH_np=np.load('/content/gdrive/My Drive/test_30clip_homograph_label_MESH_np.npy')

print(len(homograph_label_MESH_np),len(val_homo_label_MESH_np),len(test_homo_label_MESH_np))
#############################################################################
#############################################################################
###############################  part 5__ 1 ######################################
#############################################################################
############################  transfer outputs in range [0 to 1] linear  ###################
#### Train Data
max_val_H1 = np.max(homograph_label_H1_np)
min_val_H1 = np.min(homograph_label_H1_np)
#homograph_label_H1_np_sh = (homograph_label_H1_np_sh - min_val_H1) / (max_val_H1 - min_val_H1 + keras.backend.epsilon())
print('max_val_H1,min_val_H1====',max_val_H1 ,min_val_H1) 

max_val_MESH = np.max(homograph_label_MESH_np)
min_val_MESH = np.min(homograph_label_MESH_np)
#homograph_label_MESH_np_sh = (homograph_label_MESH_np_sh - min_val_MESH) / (max_val_MESH - min_val_MESH + keras.backend.epsilon())
print('max_val_MESH,min_val_MESH====',max_val_MESH,min_val_MESH)

#max_val_H1,min_val_H1 = [338.85938 , -401.8903]
#max_val_MESH,min_val_MESH = [68.732445 , -45.588158]


# ####################################### Val Data
# max_val_H1 = np.max(val_homo_label_H1_np)
# min_val_H1 = np.min(val_homo_label_H1_np)
# print('2 max_val_H1,2 min_val_H1====',max_val_H1 ,min_val_H1) 

val_homo_label_H1_np = (val_homo_label_H1_np - min_val_H1) / (max_val_H1 - min_val_H1 + keras.backend.epsilon())

# max_val_MESH = np.max(val_homo_label_MESH_np)
# min_val_MESH = np.min(val_homo_label_MESH_np)
# print('2max_val_MESH,2min_val_MESH====',max_val_MESH,min_val_MESH)
val_homo_label_MESH_np = (val_homo_label_MESH_np - min_val_MESH) / (max_val_MESH - min_val_MESH + keras.backend.epsilon())


# # 2 max_val_H1,2 min_val_H1==== 272.69824 -201.65582
# # 2max_val_MESH,2min_val_MESH==== 49.775505 -50.316322

# ####################################### test Data
# max_val_H1 = np.max(test_homo_label_H1_np)
# min_val_H1 = np.min(test_homo_label_H1_np)
# print('2 max_test_H1,2 min_val_H1====',max_test_H1 ,min_val_H1) 

test_homo_label_H1_np = (test_homo_label_H1_np - min_val_H1) / (max_val_H1 - min_val_H1 + keras.backend.epsilon())

# max_test_MESH = np.max(test_homo_label_MESH_np)
# min_test_MESH = np.min(test_homo_label_MESH_np)
# print('2max_test_MESH,2min_test_MESH====',max_test_MESH,min_test_MESH)

test_homo_label_MESH_np = (test_homo_label_MESH_np - min_val_MESH) / (max_val_MESH - min_val_MESH + keras.backend.epsilon())



print('Train max_H1,min_H1====',np.max(homograph_label_H1_np) ,np.min(homograph_label_H1_np)) 
print('Train max_MESH,min_MESH====',np.max(homograph_label_MESH_np) ,np.min(homograph_label_MESH_np))  

print('validation max_H1,min_H1====',np.max(val_homo_label_H1_np) ,np.min(val_homo_label_H1_np)) 
print('validation max_MESH,min_MESH====',np.max(val_homo_label_MESH_np) ,np.min(val_homo_label_MESH_np)) 

print('test max_H1,min_H1====',np.max(test_homo_label_H1_np) ,np.min(test_homo_label_H1_np)) 
print('test max_MESH,min_MESH====',np.max(test_homo_label_MESH_np) ,np.min(test_homo_label_MESH_np)) 

 # 5300 196 253
# max_val_H1,min_val_H1==== 338.85938 -401.8903
# max_val_MESH,min_val_MESH==== 68.732445 -45.588158
# Train max_H1,min_H1==== 338.85938 -401.8903
# Train max_MESH,min_MESH==== 68.732445 -45.588158
# validation max_H1,min_H1==== 0.6986104 0.44225082
# validation max_MESH,min_MESH==== 0.6427046 0.072089225
# test max_H1,min_H1==== 0.7621886 0.34070268
# test max_MESH,min_MESH==== 0.73651797 0.004023797   


#############################################################################################
#############################################################################################
###############################  part 6 #####################################################
#############################################################################################
######################################### VOTING  testing model by voting  ##################

MHEmodel.load_weights(pathgd+'best_model/HME23_EMO15_46_E_10_val_loss_1.1626_val_ACC_0.8387.weights.h5')
                     # Loading the best model from previous Training
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)
# dictionary which assigns each label an emotion (alphabetical order)
MHEmodel_dict = {0: "truth", 1: "lie"}


test_vid_np1 = test_vid_np[34:34+78]  ##[:] [0:34][34:34+78]  [34+78:34+78+47] [34+78+47:34+78+47+56][34+78+47+56:34+78+47+56+38]
test_vid_label_01_np1 = test_vid_label_01_np[34:34+78]
test_mic_exp_labels1 = test_mic_exp_labels[34:34+78]
test_Emo_label_np1 = test_Emo_label_np[34:34+78]
test_homo_label_H1_np1 = test_homo_label_H1_np[34:34+78]
test_homo_label_MESH_np1 = test_homo_label_MESH_np[34:34+78]


bb = test_homo_label_MESH_np1.shape[0] 
print('num of dataset ====',bb) ## num of dataset ==== 22
decept_detect_matrix = []
decept_detect_matrixb1 = []
decept_detect_matrixpred2 = []
ME_matrixpred1 = []
ME_matrixb0 = []

bb3 = int(bb/3)  ## 3 is batch size 
for kk2 in range (bb3):
    a0= test_vid_np1[0+3*kk2:3+3*kk2] ## 3 is batch size
    a1= test_Emo_label_np1[0+3*kk2:3+3*kk2] ## 3 is batch size
    a2= test_homo_label_H1_np1[0+3*kk2:3+3*kk2] ## 3 is batch size
    a3= test_homo_label_MESH_np1[0+3*kk2:3+3*kk2] ## 3 is batch size
    
    b0= test_mic_exp_labels1[0+3*kk2:3+3*kk2] ## 3 is batch size (3,107)
    b1= test_vid_label_01_np1[0+3*kk2:3+3*kk2]  ## 3 is batch size  (3,2)
    

    pred1,pred2 = MHEmodel.predict([a0,a1,a2,a3])  ##MHEmodel.predict DELETED 5 == NUM of input MHEmodel
    print ('testing model M_EXP,Decept_Detect ====',pred1.shape,pred2.shape) 

    ##pred1,pred2,pred3,loss10 = MHEmodel.predict([a0,a1,a2,a3])  ##MHEmodel.predict DELETED 5 == NUM of input MHEmodel
    ##print ('testing model M_EXP,Decept_Detect,mesh,g_loss ====',pred1.shape,pred2.shape,pred3.shape,loss10.shape) 
    ##== (3, 107) (3, 2) (3, 1) (3, 786432)
    ## (3, 107) (3, 2) ()  ##  shape=(3, 1), dtype=float32)

    #print('g_loss ====',loss10)  ## tf.Tensor(1.2502934, shape=(), dtype=float32) only one num.
    #print('mesh ====',pred3)  
    #print('M_EXP,Decept_Detect ====',pred1)
    print('Decept_Detect ====',pred2)

    maxindexpred1 = int(np.argmax(pred1[0])), int(np.argmax(pred1[1])), int(np.argmax(pred1[2])) ## batch_size === 3 
    maxindexb0 = int(np.argmax(b0[0])), int(np.argmax(b0[1])), int(np.argmax(b0[2])) ## batch_size === 3 

    maxindex = int(np.argmax(pred2[0])), int(np.argmax(pred2[1])), int(np.argmax(pred2[2])) ## batch_size === 3 
    maxindexb1 = int(np.argmax(b1[0])), int(np.argmax(b1[1])), int(np.argmax(b1[2])) ## batch_size === 3 



    print('Deception detection pred2 maxindex: ====',maxindex)
    print('Deception detection b1 maxindex: ====',maxindexb1)

    
    print('Bath nummber from all Batch : ====',kk2,'/',bb3)
    print('Deception detection outputs: ====',MHEmodel_dict[maxindex[0]],MHEmodel_dict[maxindex[1]]
        ,MHEmodel_dict[maxindex[2]])

    decept_detect_matrix.append(MHEmodel_dict[maxindex[0]])
    decept_detect_matrix.append(MHEmodel_dict[maxindex[1]])
    decept_detect_matrix.append(MHEmodel_dict[maxindex[2]])
    #########
    
    decept_detect_matrixpred2.append([maxindex[0]])
    decept_detect_matrixpred2.append([maxindex[1]])
    decept_detect_matrixpred2.append([maxindex[2]])    

    decept_detect_matrixb1.append([maxindexb1[0]])
    decept_detect_matrixb1.append([maxindexb1[1]])
    decept_detect_matrixb1.append([maxindexb1[2]])
    #######
    
    ME_matrixpred1.append([maxindexpred1[0]])
    ME_matrixpred1.append([maxindexpred1[1]])
    ME_matrixpred1.append([maxindexpred1[2]])    

    ME_matrixb0.append([maxindexb0[0]])
    ME_matrixb0.append([maxindexb0[1]])
    ME_matrixb0.append([maxindexb0[2]])

print('decept_detect_matrix =====',decept_detect_matrix)
print('decept_detect_matrixpred2 =====',decept_detect_matrixpred2)
print('decept_detect_matrixb1 =====',decept_detect_matrixb1)
print('ME_matrixpred1 pred2 =====',ME_matrixpred1)
print('ME_matrixb0 =====',ME_matrixb0)

num_lie = 0
num_truth = 0

for name in decept_detect_matrix:
        if 'lie' in name:
            num_lie = num_lie+1
        else:
            num_truth = num_truth+1

print('num_lie is =====',num_lie,'/',len(decept_detect_matrix))
print('num_truth is =====',num_truth,'/',len(decept_detect_matrix))
print('percent Of lie is === %%%%%%',100*num_lie/(num_truth+num_lie))

###########################  Deception Detection   ACC ROC

num_Dec_Detect_matching = 0
lennn = len(decept_detect_matrixpred2)
for i in range (lennn):
        if decept_detect_matrixpred2[i] == decept_detect_matrixb1[i]:
            num_Dec_Detect_matching = num_Dec_Detect_matching+1
        else:
            num_Dec_Detect_matching = num_Dec_Detect_matching

print('num_Dec_Detect_matching is =====',num_Dec_Detect_matching,'/',lennn)
print('percent Of num_Dec_Detect_matching is === %%%%%%',100*num_Dec_Detect_matching/(lennn))

###########################  Micro  EXpression  ACC
num_ME_matching = 0
lennn = len(ME_matrixpred1)
for i in range (lennn):
        if ME_matrixpred1[i] == ME_matrixb0[i]:
            num_ME_matching = num_ME_matching+1
        else:
            num_ME_matching = num_ME_matching

print('num_ME_matching is =====',num_ME_matching,'/',lennn)
print('percent Of num_ME_matching is === %%%%%%',100*num_ME_matching/(lennn))



################################################################################
##############################   part 7  #######################################
################################################################################
########################### Model.Evaluate  && Model.Predict    ##################

bb3 = test_vid_np.shape[0]  ### batc size ==== b
test_homo_out = np.zeros(bb3,).astype(np.float32)##(None,)(130,)
test_homo_out_1 = test_homo_out   ###  (b=353,) shape (353,)

test_in_data =[test_vid_np,test_Emo_label_np
                ,test_homo_label_H1_np,test_homo_label_MESH_np
                ]
# test_out_data = [test_mic_exp_labels,test_vid_label_01_np
                # ,test_homo_out,test_homo_out_1]
test_out_data = [test_mic_exp_labels,test_vid_label_01_np
                ]                

print ('test_in_data====',len(test_in_data),len(test_in_data[0]))
print ('test_out_data====',len(test_out_data),len(test_out_data[0]))

##########################################  model.evaluate
################################################################################
MHEmodel.compile(
    #optimizer=keras.optimizers.Adam(3e-4),
    optimizer = keras.optimizers.Adam(learning_rate=0.00000498), ##0.0001 0.00002466  0.0001/2#0.00001506 ## base rate=0.0001  0.00005 0.00001  ##0.00002744
    #loss=keras.losses.categorical_crossentropy,## binary_crossentropy  None  ##[M_EXP,Decept_Detect,mash,g_loss]
    #loss = [None,None,None ,'categorical_crossentropy'], ## 'mean_squared_error'
    #loss = ['categorical_crossentropy','categorical_crossentropy', 'binary_crossentropy','categorical_crossentropy'],
    #loss = ['categorical_crossentropy','categorical_crossentropy', 'binary_crossentropy','mean_squared_error'],
    #loss = ['categorical_crossentropy','categorical_crossentropy', None,None],
    loss = ['categorical_crossentropy','categorical_crossentropy'],
    #loss_weights=[.3, .7, .0, .0],  ##[.2, .6, .1, .1]
    loss_weights=[.4, .6],  ##
    
    ### ACC ###
    #metrics=[None,None,None,'accuracy']
    #metrics=['accuracy','accuracy','accuracy','accuracy']
    #metrics=['categorical_accuracy','categorical_accuracy','binary_accuracy','binary_accuracy'],
    #metrics=[None,'categorical_accuracy',None,None],
    metrics=['categorical_accuracy','categorical_accuracy'],
    
    ####  ROC_AUC ####
    #metrics=[keras.metrics.AUC(),'categorical_accuracy',None,None]
    #metrics=[keras.metrics.AUC(multi_label=True, num_labels=107),keras.metrics.AUC(multi_label=True, num_labels=2),None,None]
    ##metrics=[keras.metrics.AUC(multi_label=True, num_labels=107),keras.metrics.AUC(multi_label=True, num_labels=2)]
    
    #weighted_metrics= [['accuracy'], ['mse']]) ,
    #weighted_metrics= [[['categorical_accuracy'], ['categorical_accuracy'] , ['binary_accuracy'] , ['binary_accuracy']])] ,
    )
    
    
score  = MHEmodel.evaluate(test_in_data, test_out_data, batch_size=3,verbose=1)
print('MHEmodel.evaluate :', score)
#print('Test loss:', score[0]) 
#print('Test accuracy:', score[1])


########################################## model.predict
bb4 = test_homo_label_MESH_np.shape[0] ## 253
bb5 = int(bb4/3) * 3  ## 252  batch_size = 3
print(bb4,bb5)
# bb5 = 6

xxxx = [test_in_data[0][0:bb5],test_in_data[1][0:bb5],test_in_data[2][0:bb5],test_in_data[3][0:bb5]]
#xxxx = [test_in_data[0][0:3],test_in_data[1][0:3],test_in_data[2][0:3],test_in_data[3][0:3]]

#pred1,pred2,pred3,loss10 = MHEmodel.predict(xxxx,batch_size = 3,verbose=1)
pred1,pred2 = MHEmodel.predict(xxxx,batch_size = 3,verbose=1) 
 
print(pred1.shape)
print(pred2.shape)

DDpred = np.argmax(pred2, axis = 1)[:5]  ## [:]
DDlabel = np.argmax(test_out_data[1],axis = 1)[:5]  ## [:] 
print('DDpred === ' , DDpred,len(DDpred)) 
print('DDlabel ==== ', DDlabel)

MEpred = np.argmax(pred1, axis = 1)[:5]  ## [:]
MElabel = np.argmax(test_out_data[0],axis = 1)[:5]  ## [:]
print('MEpred == ', MEpred,len(MEpred)) 
print('MElabel === ', MElabel)


##################################################################################
##################################################################################
############################ logging  LOGGING  ###################################
##################################################################################

#########################################
#########################################
#########################################   NEW ROC 
dense_4 (Dense)     │ (None, 2)         │      2,050 │ dropout_6[0][0]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (None, 2)         │          8 │ dense_4[0][0]     │
│ (BatchNormalizatio… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ Decept_Detect       │ (None, 2)         │          0 │ batch_normalizat… │
│ (Activation)        │                   │            │                   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 33,884,065 (129.26 MB)
 Trainable params: 33,877,319 (129.23 MB)
 Non-trainable params: 6,746 (26.35 KB)
len(MHEmodel.layers)=== 42

5300 196 253
max_val_H1,min_val_H1==== 338.85938 -401.8903
max_val_MESH,min_val_MESH==== 68.732445 -45.588158
Train max_H1,min_H1==== 338.85938 -401.8903
Train max_MESH,min_MESH==== 68.732445 -45.588158
validation max_H1,min_H1==== 0.6986104 0.44225082
validation max_MESH,min_MESH==== 0.6427046 0.072089225
test max_H1,min_H1==== 0.7621886 0.34070268
test max_MESH,min_MESH==== 0.73651797 0.004023797

decept_detect_matrix ===== ['lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie']
decept_detect_matrixpred2 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
decept_detect_matrixb1 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
ME_matrixpred1 pred2 ===== [[93], [84], [93], [84], [84], [83], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84]]
ME_matrixb0 ===== [[102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102]]
num_lie is ===== 33 / 33
num_truth is ===== 0 / 33
percent Of lie is === %%%%%% 100.0
num_Dec_Detect_matching is ===== 33 / 33
percent Of num_Dec_Detect_matching is === %%%%%% 100.0
num_ME_matching is ===== 0 / 33
percent Of num_ME_matching is === %%%%%% 0.0

Deception detection outputs: ==== lie lie lie
decept_detect_matrix ===== ['lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie']
decept_detect_matrixpred2 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
decept_detect_matrixb1 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
ME_matrixpred1 pred2 ===== [[70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70]]
ME_matrixb0 ===== [[70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70]]
num_lie is ===== 78 / 78
num_truth is ===== 0 / 78
percent Of lie is === %%%%%% 100.0
num_Dec_Detect_matching is ===== 78 / 78
percent Of num_Dec_Detect_matching is === %%%%%% 100.0
num_ME_matching is ===== 78 / 78
percent Of num_ME_matching is === %%%%%% 100.0

decept_detect_matrix ===== ['lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie']
decept_detect_matrixpred2 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
decept_detect_matrixb1 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
ME_matrixpred1 pred2 ===== [[70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70]]
ME_matrixb0 ===== [[103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103]]
num_lie is ===== 45 / 45
num_truth is ===== 0 / 45
percent Of lie is === %%%%%% 100.0
num_Dec_Detect_matching is ===== 45 / 45
percent Of num_Dec_Detect_matching is === %%%%%% 100.0
num_ME_matching is ===== 0 / 45
percent Of num_ME_matching is === %%%%%% 0.0


decept_detect_matrix ===== ['lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie']
decept_detect_matrixpred2 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
decept_detect_matrixb1 ===== [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
ME_matrixpred1 pred2 ===== [[17], [17], [17], [77], [81], [81], [81], [17], [17], [17], [17], [81], [81], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [81], [81], [81], [17], [17], [81], [17], [17], [17], [17], [17], [17], [17], [17], [81], [17], [17], [17]]
ME_matrixb0 ===== [[51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51]]
num_lie is ===== 54 / 54
num_truth is ===== 0 / 54
percent Of lie is === %%%%%% 100.0
num_Dec_Detect_matching is ===== 0 / 54
percent Of num_Dec_Detect_matching is === %%%%%% 0.0
num_ME_matching is ===== 0 / 54
percent Of num_ME_matching is === %%%%%% 0.0

Deception detection outputs: ==== truth truth truth
decept_detect_matrix ===== ['lie', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'lie', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth']
decept_detect_matrixpred2 ===== [[1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
decept_detect_matrixb1 ===== [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
ME_matrixpred1 pred2 ===== [[28], [28], [78], [51], [28], [28], [51], [51], [51], [51], [51], [51], [28], [28], [28], [28], [28], [28], [28], [28], [25], [25], [28], [51], [28], [51], [51], [51], [51], [51], [28], [28], [51], [51], [28], [28]]
ME_matrixb0 ===== [[48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48]]
num_lie is ===== 2 / 36
num_truth is ===== 34 / 36
percent Of lie is === %%%%%% 5.555555555555555
num_Dec_Detect_matching is ===== 34 / 36
percent Of num_Dec_Detect_matching is === %%%%%% 94.44444444444444
num_ME_matching is ===== 0 / 36
percent Of num_ME_matching is === %%%%%% 0.0


#################################################################

decept_detect_matrix ===== ['lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'lie', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'lie', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth', 'truth']
decept_detect_matrixpred2 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
decept_detect_matrixb1 ===== [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
ME_matrixpred1 pred2 ===== [[93], [84], [93], [84], [84], [83], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [84], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [17], [17], [17], [77], [81], [81], [81], [17], [17], [17], [17], [81], [81], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [17], [81], [81], [81], [17], [17], [81], [17], [17], [17], [17], [17], [17], [17], [17], [81], [17], [17], [17], [17], [17], [28], [28], [78], [51], [28], [28], [51], [51], [51], [51], [51], [51], [28], [28], [28], [28], [28], [28], [28], [28], [25], [25], [28], [51], [28], [51], [51], [51], [51], [51], [28], [28], [51], [51], [28], [28], [28]]
ME_matrixb0 ===== [[102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [102], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [70], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [103], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [51], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48], [48]]
num_lie is ===== 217 / 252
num_truth is ===== 35 / 252
percent Of lie is === %%%%%% 86.11111111111111
num_Dec_Detect_matching is ===== 194 / 252
percent Of num_Dec_Detect_matching is === %%%%%% 76.98412698412699
num_ME_matching is ===== 78 / 252
percent Of num_ME_matching is === %%%%%% 30.952380952380953

#######################################################################


test_in_data==== 4 253
test_out_data==== 2 253
85/85 ━━━━━━━━━━━━━━━━━━━━ 107s 1s/step - Decept_Detect_categorical_accuracy: 0.9268 - Decept_Detect_loss: 0.1971 - M_EXP_categorical_accuracy: 0.3994 - M_EXP_loss: 2.9997 - loss: 1.3180
MHEmodel.evaluate : [1.5124578475952148, 3.3028292655944824, 0.3258633613586426, 0.7707509994506836, 0.3083004057407379]
253 252
84/84 ━━━━━━━━━━━━━━━━━━━━ 82s 969ms/step
(252, 107)
(252, 2)
DDpred ===  [1 1 1 1 1] 5
DDlabel ====  [1 1 1 1 1]
MEpred ==  [93 84 93 84 84] 5
MElabel ===  [102 102 102 102 102]

###########################################################################################################################
###########################################################################################################################
