import os
import time
import cv2
import numpy as np
import pipeline
import monitoring
import constants
import GUI
import signal
import Mode
import notify_user

source_mode = True #True = camera and False = video
video_path = 0


f = open("videopath.txt", "w")
f.close()
os.system("rm videopath.txt")

def isq(signum, frame):
    global source_mode, video_path
    f = open("videopath.txt", "r")
    video_path = f.read()
    if os.path.isfile(video_path) == 0:
        video_path = 0
    else:
        source_mode = False

signal.signal(signal.SIGALRM, isq)



mode_pid = os.fork()
if mode_pid == 0:
    Mode.run()
    os._exit(0)
else:
    while True:
        childProcExitInfo = os.waitpid(mode_pid, os.WNOHANG)
        if childProcExitInfo[0] == mode_pid:
            break

is_running = False

def isr(signum, frame):
    global is_running
    if is_running == True:
        is_running = False
    else:
        is_running = True

    

signal.signal(signal.SIGALRM, isr)


gui_pid = os.fork()
if gui_pid == 0:
    GUI.run()
    is_running = False
    os._exit(0)
else:
    while is_running == False:
        childProcExitInfo = os.waitpid(gui_pid, os.WNOHANG)
        if childProcExitInfo[0] == gui_pid:
            quit()
    


# Constants
FPS = constants.FPS
UPS = constants.UPS
SPF = constants.SPF

"""
# Remove previous videos
os.system("rm videos/*")
os.system("rm compressed/*")

# Remove previous out
os.system("rm -rf Anomaly-Detection/out/*")
"""

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Variables
path = ""
global_path = "video" + str(int(time.time()))
frames_cnt = 0
newpid = 0

outpid = os.fork()
if outpid == 0:
    monitoring.run(global_path, frame_width, frame_height)
    os._exit(0)

# Prediction
def child():
    if frames_cnt == 0:
	    time.sleep(0)
    else:
        pipeline.pipeline(path, frames_cnt)

frame_cnt_all = 0

# Read from Camera and Store
id = 1
cont_next_time = 1
def parent():
    global frames_cnt, path, id, frame_cnt_all, is_running, cont_next_time
    if cont_next_time == 0:
        is_running = 0
        return

    frames_cnt = 0

    path = global_path + str(id) + ".mp4"

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter("videos/" + path,cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
    
    can_break = 0
    last_frame = None
    while can_break == 0 or frames_cnt % (FPS * UPS * SPF) != 0:
        ret, frame = cap.read()
        
        if can_break == 0:
            childProcExitInfo = os.waitpid(newpid, os.WNOHANG)
            if childProcExitInfo[0] == newpid:
                can_break = 1

        if ret == False:
            if frames_cnt == 0:
                is_running = False
                break
            if source_mode == False:
                frame = last_frame
                cont_next_time = False
                if frames_cnt % (FPS * UPS * SPF) == 0:
                    continue

        # Write the frame
        out.write(frame)
        frames_cnt += 1
        frame_cnt_all += 1
        
        cv2.putText(img = frame, 
            text = str(frame_cnt_all),
            org = (constants.MARGIN_W_P, constants.MARGIN_H_P + 20), 
            fontFace = cv2.FONT_HERSHEY_DUPLEX, 
            fontScale = constants.FONT_SCALE_P, 
            color = constants.NORMAL_COLOR,
            thickness = constants.FONT_THICKNESS_P, 
            lineType = cv2.LINE_AA)

        
        cv2.imshow('original', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if source_mode == False:
            time.sleep(constants.ACCELERATING_MONITORING_FACTOR)
        
        last_frame = frame

    print(frames_cnt)
    id += 1
    out.release()


gui_running = True
out_running = True

while is_running:
    newpid = os.fork()
    if newpid == 0:
        child()
        os._exit(0)
    else:   
        parent()
    childProcExitInfo = os.waitpid(outpid, os.WNOHANG)
    if childProcExitInfo[0] == outpid:
        is_running = False
        out_running = False
        print("monitor stopped")
    childProcExitInfo = os.waitpid(gui_pid, os.WNOHANG)
    if childProcExitInfo[0] == gui_pid:
        is_running = False
        gui_running = False        
        print("gui stopped")

"""
while True:
    childProcExitInfo = os.waitpid(newpid, os.WNOHANG)
    if childProcExitInfo[0] == newpid:
        newpid = os.fork()
        if newpid == 0:
            child()
            os._exit(0)
        break
"""

cv2.destroyAllWindows()
"""
if out_running:
    os.kill(outpid, signal.SIGALRM)
"""
if gui_running:
    os.kill(gui_pid, signal.SIGALRM)
while True:
    if out_running:
        childProcExitInfo = os.waitpid(outpid, os.WNOHANG)
        if childProcExitInfo[0] == outpid:
            out_running = False
    if gui_running:
        childProcExitInfo = os.waitpid(gui_pid, os.WNOHANG)
        if childProcExitInfo[0] == gui_pid:
            gui_running = False
    if gui_running == 0 and out_running == 0:
        break
cap.release()
print("Source Ended")
quit()



"""
killing_pid = os.fork()
if killing_pid == 0:
    time.sleep(5)
    #os.system("kill -9 " + str(os.getppid()))
    os.system("kill -9 " + str(outpid))
    #os.system("kill -9 " + str(gui_pid))
    os._exit(0)
else:
    cap.release()
"""
