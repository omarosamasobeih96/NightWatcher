import os
import cv2
import time
import constants

# Constants
FPS = constants.FPS
UPS = constants.UPS
SPF = constants.SPF


def read_video(path):
    return cv2.VideoCapture(path)

def calc_prediction_path(path, cur, idx):
    return path + str(cur) + "/" + str(idx) + "_C.mat"


def read_predictions(path, cur, files_cnt):
    predictions = []
    is_anomaly = []
    for i in range (0, files_cnt):
        readings = open(calc_prediction_path(path, cur, i), "r")
        for j in range (0, SPF):
            cur_reading = float(readings.readline())
            predictions.append(cur_reading)
            is_anomaly.append(cur_reading > constants.THRESHOLD)
    return predictions, is_anomaly

def check_prediction_files_exist(path, cur, files_cnt):
    for i in range (0, files_cnt):
        cur_path_out = calc_prediction_path(path, cur, i)
        if os.path.isfile(cur_path_out) == 0:
            print("error loading prediction files")
            return 0
    return 1



def run(path, frame_width, frame_height):

    """
    while True:
        time.sleep(10)
    """

    path_vid = "videos/" + path
    path_out = "Anomaly-Detection/out/" + path

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter("compressed/" + path + ".mp4" ,cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))

    
    cur = 1
    lets_exit = 0
    lst_clc = 0

    while lets_exit == 0:

        cur_path_out_0 = calc_prediction_path(path_out, cur, 0)
        while os.path.isfile(cur_path_out_0) == 0:
            time.sleep(4)

        cap = read_video(path_vid + str(cur) + ".mp4")
        
        frames_cnt = cap.get(7)
        frame_width = cap.get(3)
        frame_height = cap.get(4)        

        units_cnt = int((frames_cnt + FPS - 1) // FPS)
        segments_cnt = int((units_cnt + UPS - 1) // UPS)
        files_cnt = int((segments_cnt + SPF - 1) // SPF)

        if check_prediction_files_exist(path_out, cur, files_cnt) == 0:
            break
        
        predictions, is_anomaly = read_predictions(path_out, cur, files_cnt)
        cnt_seg = 0
        cnt_frm = 0
        prnted = ""
        color_txt = (0,0,0)
        prv_seg = -1
        all_seg = len(is_anomaly)         
        propis = 0

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == False:
                break

            if all_seg <= cnt_seg:
                print ("predictions and video length don't match")
                break

            if prv_seg != cnt_seg:
                prv_seg = cnt_seg
                cur1 = is_anomaly[cnt_seg]
                prev = cur1
                nxt = cur1
                if cnt_seg != 0:
                    prev = is_anomaly[cnt_seg - 1]
                if cnt_seg + 1 < len(predictions):
                    nxt = is_anomaly[cnt_seg + 1]
                if nxt != cur1 and prev != cur1:
                    is_anomaly[cnt_seg] = prev

                prnted = constants.NORMAL_TEXT
                color_txt = constants.NORMAL_COLOR
                propis = "Prob :  " + str(int(100 * (1 - predictions[cnt_seg]))) + "%"
                if is_anomaly[cnt_seg]:
                    prnted = constants.ANOMALY_TEXT
                    color_txt = constants.ANOMALY_COLOR
                    propis = "Prob :  " + str(int(100 * predictions[cnt_seg])) + "%"

            if is_anomaly[cnt_seg]:
                out.write(frame)

            cv2.putText(img = frame, 
                text = prnted,
                org = (constants.MARGIN_W, constants.MARGIN_H), 
                fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                fontScale = constants.FONT_SCALE, 
                color = color_txt,
                thickness = constants.FONT_THICKNESS, 
                lineType = cv2.LINE_AA)

            cv2.putText(img = frame, 
                text = propis,
                org = (constants.MARGIN_W_P, constants.MARGIN_H_P), 
                fontFace = cv2.FONT_HERSHEY_DUPLEX, 
                fontScale = constants.FONT_SCALE_P, 
                color = color_txt,
                thickness = constants.FONT_THICKNESS_P, 
                lineType = cv2.LINE_AA)

            wait_time = max(0, (1/FPS) - time.time() + lst_clc - constants.ACCELERATING_MONITORING_FACTOR)
            time.sleep(wait_time)

            cv2.imshow('monitoring', frame)

            lst_clc = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                lets_exit = 1
                break

            #time.sleep(1/(constants.ACCELERATING_MONITORING_FACTOR*FPS))

            cnt_frm += 1
            if cnt_frm % (FPS * UPS) == 0:
                cnt_seg += 1


        cur += 1

    out.release()

    cv2.destroyAllWindows()