import os
import cv2
import time

FPS = 16    # frames per second


def read_video(path):
    return cv2.VideoCapture(path)


def read_predictions(path):
    return ""

def run(path):
    path_vid = "videos/" + path
    path_out = "Anomaly-Detection/out/" + path
    
    cur = 1
    lets_exit = 0

    while lets_exit == 0:

        cur_path_out_0 = path_out + str(cur) + "/0_C.mat"

        print(cur_path_out_0)

        while os.path.isfile(cur_path_out_0) == 0:
            time.sleep(4)



        cap = read_video(path_vid + str(cur) + ".mp4")
        frames_cnt = cap.get(7)
        print("frames cnt equals: " + str(frames_cnt))
        
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == False:
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                lets_exit = 1
                break

            time.sleep(1/FPS)
        # predictions = read_predictions(path_out + str(cur))
        cur += 1

    cv2.destroyAllWindows()

