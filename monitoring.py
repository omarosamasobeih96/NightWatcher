import os
import cv2
import time

def read_video(path):
    print(path)
    cap = cv2.VideoCapture(path)
    while cap is None or not cap.isOpened():
        time.sleep(4)
        cap = cv2.VideoCapture(path)
    return cap


def read_predictions(path):
    return ""

def run(path):
    path_vid = "videos/" + path
    #path_out = "Anomaly-Detection/out/" + path
    
    cur = 1

    while True:
        cap = read_video(path_vid + str(cur) + ".mp4")
        frames_cnt = cap.get(7)
        cap.release()
        print("frames cnt equals: " + str(frames_cnt))

        """
        
        while(cap.isOpened()):
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        """
        
        # predictions = read_predictions(path_out + str(cur))
        cur += 1

    cv2.destroyAllWindows()

