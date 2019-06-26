#prepare input and output lists for model which extracts featurs
 
import cv2
import os
import random

videos_path = "videos/"
output_path = "Anomaly-Detection/C3D_Features"
avg_path = "Anomaly-Detection/C3D_Features_Avg"
prototxt_path = "C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt"

in_list_path = "input_list_video.txt"
out_list_path = "output_list_video_prefix.txt"
num_vid_in_seg = 32

in_list_path = os.path.join(prototxt_path , in_list_path)
out_list_path = os.path.join(prototxt_path , out_list_path)


def readVideo(video , len):
    cur_video_path = os.path.join(videos_path , video)
    in_list = open(in_list_path, "w+")
    out_list = open(out_list_path, "w+")
    make_dir_script = open("mk.sh", "w+")
    make_dir_avg_script = open("mk1.sh", "w+")
    cnt = 0
    cur_seg = 0
    make_dir_avg_script.write("mkdir -p " + os.path.join(avg_path , video[0:-4]) + "\n" )
    make_dir_script.write("mkdir -p " +  os.path.join(os.path.join(output_path , os.path.join(video[0:-4],str(cur_seg)))) + "\n")
    sum_cur_seg = 0
    for st_frame in range(0 , len , 16):
        if st_frame + 16 >= len:
          break
        cnt+=1
        in_list.write(cur_video_path + " " + str(st_frame) + " 0\n")
        out_list.write(os.path.join(os.path.join(output_path , os.path.join(video[0:-4],str(cur_seg))) , str(st_frame)) + "\n")
        sum_cur_seg += 1
        if sum_cur_seg == num_vid_in_seg:
            cur_seg += 1
            sum_cur_seg = 0
            make_dir_script.write("mkdir -p " + os.path.join(os.path.join(output_path, os.path.join(video[0:-4], str(cur_seg)))) + "\n")
    in_list.close()
    out_list.close()
    make_dir_script.close()
    make_dir_avg_script.close()
    return cnt


def preprocessing(video , len):

    tot = readVideo(video , len)
    print(tot)

    batch_size = 10
    num_of_batches = (tot + batch_size - 1) // batch_size
    f = open("C3D/C3D-v1.0/examples/c3d_feature_extraction/c3d_sport1m_feature_extraction_video.sh", "w+")
    f.write(
        "GLOG_logtosterr=1 C3D/C3D-v1.0/build/tools/extract_image_features.bin  C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt C3D/C3D-v1.0/conv3d_deepnetA_sport1m_iter_1900000 0 10 ")
    f.write(str(num_of_batches))
    f.write(" C3D/C3D-v1.0/examples/c3d_feature_extraction/prototxt/output_list_video_prefix.txt fc6-1")
    f.close()
