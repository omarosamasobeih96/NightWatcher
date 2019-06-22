# This code save already computed C3D features into 32 (video features) segments.
# We assume that C3D features for a video are already computed. We use default settings for computing C3D features, 
# i.e., we compute C3D features for every 16 frames and obtain the features from fc6.

import numpy as np
import os
import array


def readBinary(path):
    f = open(path, 'rb')
    s = array.array("i") # int32
    s.fromfile(f, 5)
    m = s[0]*s[1]*s[2]*s[3]*s[4]
    data_aux = array.array("f")
    data_aux.fromfile(f, m)
    data = np.array(data_aux.tolist())
    return s,data



C3D_path='Anomaly-Detection/C3D_Features'
C3D_path_seg='Anomaly-Detection/C3D_Features_Avg'

if not os.path.exists(C3D_path_seg):
    os.makedirs(C3D_path_seg)

all_folders = os.listdir(C3D_path)
subcript='_C.txt'


x = 1
for ifolder in all_folders:
    folder_path = os.path.join(C3D_path , ifolder) 
    all_seg_folders = os.listdir(folder_path)
    
    for seg_folder in all_seg_folders:
        seg_path = os.path.join(folder_path , seg_folder) 
        all_files = os.listdir(seg_path)
        all_files_int = []
        for i in range(len(all_files)):
           all_files_int.append(int(all_files[i][0:-6]))
        all_files_int.sort()
        all_files = []
        for i in range(len(all_files_int)):
           all_files.append(str(all_files_int[i]) + ".fc6-1")

        feature_vect = np.zeros((len(all_files) , 4096))
        for ifile in range(len(all_files)):
            file_path  = os.path.join(seg_path , all_files[ifile])
            s , data = readBinary(file_path)
            feature_vect[ifile] = data


        if np.sum(feature_vect) == 0:
            print("all data are zeros" + " " + seg_path)
            continue

        if np.sum( np.sum(feature_vect , 1) == 0  ):
            print("some rows are zeros")
            continue

        if np.isnan(feature_vect).any():
            print("some values are missing")
            continue

        if np.isinf(feature_vect).any():
            print("some values are inf")
            continue


        # Now all is okay, time to store the features for 32 segments

        segment_features = np.zeros((32,4096))
        positions= np.round(np.linspace(0,len(all_files)-1,33))

        for iposition in range(len(positions) - 1):
            cur = int(positions[iposition])
            nxt = int(positions[iposition + 1])-1

            if cur >= nxt:
                temp_vect = feature_vect[cur]
            else:
                temp_vect = np.max(feature_vect[cur:nxt+1] , axis = 0)

            nrm = np.linalg.norm(temp_vect)

            if nrm == 0:
                print("normalization is wrong")
                exit()

            temp_vect=temp_vect/nrm
            segment_features[iposition] = temp_vect

        # save features file
        np.savetxt(os.path.join(C3D_path_seg , os.path.join(ifolder , seg_folder + subcript)) , segment_features , fmt="%f")
