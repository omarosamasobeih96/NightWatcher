from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import configparser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import theano.sandbox
import shutil
num_abnormal = 1
num_normal = 1
def saveModel(model, json_path, weight_path): # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)

batch_size=min(60 , (num_abnormal + num_normal) // 2)
# Load Training Dataset

def loadDataTrain(all_class_path):
#    print("Loading training batch")

    n_exp=batch_size//2  # Number of abnormal and normal videos


    # We assume the features of abnormal videos and normal videos are located in two different folders.
    abnor_list_iter = np.random.permutation(num_abnormal)
    abnor_list_iter = abnor_list_iter[num_abnormal-n_exp:] # Indexes for randomly selected Abnormal Videos
    norm_list_iter = np.random.permutation(num_normal)
    norm_list_iter = norm_list_iter[num_normal-n_exp:]     # Indexes for randomly selected Normal Videos

     
    all_Videos=listdir(all_class_path)
    all_Videos.sort()
    
    anomaly_videos = []
    normal_videos = []
    for video in all_Videos:
        if video[0] == 'N':
            normal_videos.append(video)
        else:
            anomaly_videos.append(video)


    all_features = []  # To store C3D features of a batch
    print("Loading Abnormal videos Features...")

    video_count=-1
    for iv in abnor_list_iter:
        video_count=video_count+1
        VideoPath = os.path.join(all_class_path, anomaly_videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        num_feat = len(words) // 4096
        # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
        # we have already computed C3D features for the whole video and divide the video features into 32 segments. Please see Save_C3DFeatures_32Segments.m as well

        count = -1;
        video_featues = []
        for feat in range(0, num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                video_featues = feat_row1
            if count > 0:
                video_featues = np.vstack((video_featues, feat_row1))

        if video_count == 0:
            all_features = video_featues
        if video_count > 0:
            all_features = np.vstack((all_features, video_featues))
        print(" Abnormal Features  loaded")

        
        
    print("Loading Normal videos...")

    for iv in norm_list_iter:
        VideoPath = os.path.join(all_class_path, normal_videos[iv])
        f = open(VideoPath, "r")
        words = f.read().split()
        feat_row1 = np.array([])
        num_feat = len(words) //4096   # Number of features to be loaded. In our case num_feat=32, as we divide the video into 32 segments.

        count = -1;
        video_featues = []
        for feat in range(0, num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
            count = count + 1
            if count == 0:
                video_featues = feat_row1
            if count > 0:
                video_featues = np.vstack((video_featues, feat_row1))
            feat_row1 = []
        all_features = np.vstack((all_features, video_featues))

    print("Features  loaded")


    all_labels = np.zeros(32*batch_size, dtype='uint8')
    th_loop1=n_exp*32
    th_loop2=n_exp*32-1

    for iv in range(0, 32*batch_size):
            if iv< th_loop1:
                all_labels[iv] = int(0)  # All instances of abnormal videos are labeled 0.  This will be used in custom_objective to keep track of normal and abnormal videos indexes.
            if iv > th_loop2:
                all_labels[iv] = int(1)   # All instances of Normal videos are labeled 1. This will be used in custom_objective to keep track of normal and abnormal videos indexes.
           # print("ALLabels  loaded")

    return  all_features,all_labels


def custom_objective(y_true, y_pred):
    'Custom Objective function'
    print(y_true)
    print(y_pred)
    y_true = T.reshape(y_true, [-1])
    y_pred = T.reshape(y_pred, [-1])

    n_seg = 32  # Because we have 32 segments per video.
    nvid = 60
    n_exp = nvid / 2
    Num_d=32*nvid


    sub_max = T.ones_like(y_pred) # sub_max represents the highest scoring instants in bags (videos).
    sub_sum_labels = T.ones_like(y_true) # It is used to sum the labels in order to distinguish between normal and abnormal videos.
    sub_sum_l1=T.ones_like(y_true)  # For holding the concatenation of summation of scores in the bag.
    sub_l2 = T.ones_like(y_true) # For holding the concatenation of L2 of score in the bag.

    for ii in xrange(0, nvid, 1):
        # For Labels
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = T.concatenate([sub_sum_labels, T.stack(T.sum(mm))])  # Just to keep track of abnormal and normal vidoes

        # For Features scores
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = T.concatenate([sub_max, T.stack(T.max(Feat_Score))])         # Keep the maximum score of scores of all instances in a Bag (video)
        sub_sum_l1 = T.concatenate([sub_sum_l1, T.stack(T.sum(Feat_Score))])   # Keep the sum of scores of all instances in a Bag (video)

        z1 = T.ones_like(Feat_Score)
        z2 = T.concatenate([z1, Feat_Score])
        z3 = T.concatenate([Feat_Score, z1])
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = T.sum(T.sqr(z))
        sub_l2 = T.concatenate([sub_l2, T.stack(z)])


    # sub_max[Num_d:] means include all elements after Num_d.
    # all_labels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[4:]
    #[  6.  12.   7.  18.   9.  14.]

    sub_score = sub_max[Num_d:]  # We need this step since we have used T.ones_like
    F_labels = sub_sum_labels[Num_d:] # We need this step since we have used T.ones_like
    #  F_labels contains integer 32 for normal video and 0 for abnormal videos. This because of labeling done at the end of "load_dataset_Train_batch"



    # all_labels =[2 , 4, 3 ,9 ,6 ,12,7 ,18 ,9 ,14]
    # z=x[:4]
    # [ 2 4 3 9]... This shows 0 to 3 elements

    sub_sum_l1 = sub_sum_l1[Num_d:] # We need this step since we have used T.ones_like
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]         # We need this step since we have used T.ones_like
    sub_l2 = sub_l2[:n_exp]


    indx_nor = theano.tensor.eq(F_labels, 32).nonzero()[0]  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
    indx_abn = theano.tensor.eq(F_labels, 0).nonzero()[0]

    n_Nor=n_exp

    Sub_Nor = sub_score[indx_nor] # Maximum Score for each of abnormal video
    Sub_Abn = sub_score[indx_abn] # Maximum Score for each of normal video

    z = T.ones_like(y_true)
    for ii in xrange(0, n_Nor, 1):
        sub_z = T.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = T.concatenate([z, T.stack(T.sum(sub_z))])

    z = z[Num_d:]  # We need this step since we have used T.ones_like
    z = T.mean(z, axis=-1) +  0.00008*T.sum(sub_sum_l1) + 0.00008*T.sum(sub_l2)  # Final Loss f

    return z

def training():  
  
    print("Create Model")
    model = Sequential()
    model.add(Dense(512, input_dim=4096,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))

    adagrad=Adagrad(lr=0.01, epsilon=1e-08)

    model.compile(loss="binary_crossentropy", optimizer=adagrad)

    print("Starting training...")

    all_class_path='Anomaly-Detection/C3D_Features_Avg'
    # all_class_path contains C3D features (.txt file)  of each video. Each text file contains 32 features, each of 4096 dimension
    output_dir='Anomaly-Detection/'
    # Output_dir is the directory where you want to save trained weights
    weights_path = output_dir + 'weights1.mat'
    # weights.mat are the model weights that you will get after (or during) that training
    model_path = output_dir + 'model1.json'

    if not os.path.exists(output_dir):
           os.makedirs(output_dir)

    all_class_files= listdir(all_class_path)
    all_class_files.sort()
    loss_graph =[]
    num_iters = 1000
    total_iterations = 0
    time_before = datetime.now()

    for it_num in range(num_iters):

        inputs, targets= loadDataTrain(all_class_path)  # Load normal and abnormal video C3D features
        batch_loss =model.train_on_batch(inputs, targets)
        loss_graph = np.hstack((loss_graph, batch_loss))
        total_iterations += 1
        if total_iterations % 20 == 1:
            print( "These iteration=" + str(total_iterations) + ") took: " + str(datetime.now() - time_before) + ", with loss of " + str(batch_loss) )
            iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
            #savemat(iteration_path, dict(loss_graph=loss_graph))
        if total_iterations % 1000 == 0:  # Save the model at every 1000th iterations.
           weights_path = output_dir + 'weightsAnomalyL1L2_' + str(total_iterations) + '.mat'
           saveModel(model, model_path, weights_path)


    saveModel(model, model_path, weights_path)

def loadModel(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model

def loadWeights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = convDict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def convDict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict

# Load Video

def loadDataTest(test_video_path):

    VideoPath = test_video_path
    f = open(VideoPath, "r")
    words = f.read().split()
    num_feat = int(len(words) / 4096)
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
    # we have already computed C3D features for the whole video and divided the video features into 32 segments.

    count = -1;
    VideoFeatues = []
    for feat in range(0, num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        count = count + 1
        if count == 0:
            VideoFeatues = feat_row1
        if count > 0:
            VideoFeatues = np.vstack((VideoFeatues, feat_row1))
    AllFeatures = VideoFeatues

    return  AllFeatures


def testing():
    seed = 7
    np.random.seed(seed)
        
    print("Starting Testing...")
    time_before = datetime.now()
    
    all_test_folder_path = 'Anomaly-Detection/C3D_Features_Avg'
    results_path = 'Anomaly-Detection/out/'
    model_dir='Anomaly-Detection/'
    weights_path = model_dir + 'weights_L1L2.mat'
    model_path = model_dir + 'model.json'

    all_test_folders= listdir(all_test_folder_path)

    model=loadModel(model_path)
    loadWeights(model, weights_path)
    
    for folder in all_test_folders:
        folder_path = os.path.join(all_test_folder_path , folder)
        all_test_files = listdir(folder_path)
        nVideos=len(all_test_files)
        for iv in range(nVideos):
            test_video_path = os.path.join(folder_path, all_test_files[iv])
            inputs=loadDataTest(test_video_path) # 32 segments features for one testing video
            predictions = model.predict_on_batch(inputs)   # Get anomaly prediction for each of 32 video segments.
            aa=all_test_files[iv]
            aa=aa[0:-4]
            out_path = os.path.join(results_path , folder)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            np.savetxt(out_path + '/' + aa + '.mat' , predictions)
            print ("Total Time took: " + str(datetime.now() - time_before))




# training()

testing()
