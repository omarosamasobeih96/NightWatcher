import os
import  preprocessing

def pipeline(video , len):
    preprocessing.preprocessing(video , len)
    os.system("rm -rf Anomaly-Detection/C3D_Features")
    os.system("sh mk.sh")
    os.system("sh C3D/C3D-v1.0/examples/c3d_feature_extraction/c3d_sport1m_feature_extraction_video.sh")
    os.system("rm -rf Anomaly-Detection/C3D_Features_Avg")
    os.system("sh mk1.sh")
    os.system("python convert.py")
    os.system("python run.py")
    print("Pipeline ended")


