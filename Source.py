import os
import time
import cv2
import numpy as np
import pipeline
import monitoring
import constants

# Constants
FPS = constants.FPS
UPS = constants.UPS
SPF = constants.SPF

# Remove previous videos
os.system("rm videos/*")
os.system("rm compressed/*")

# Remove previous out
os.system("rm -rf Anomaly-Detection/out/*")

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

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
	    time.sleep(5)
    else:
        pipeline.pipeline(path, frames_cnt)

# Read from Camera and Store
id = 1
def parent():
    global frames_cnt, path, id
    frames_cnt = 0

    path = global_path + str(id) + ".mp4"

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter("videos/" + path,cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
    
    can_break = 0

    while can_break == 0 or frames_cnt % (FPS * UPS * SPF) != 0:
        ret, frame = cap.read()
        
        # Write the frame
        if ret != True:
            os._exit(0)
        
        cv2.imshow('original', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)
        frames_cnt = frames_cnt + 1
    
        if can_break == 0:
            childProcExitInfo = os.waitpid(newpid, os.WNOHANG)
            if childProcExitInfo[0] == newpid:
                can_break = 1


    id += 1
    out.release()

while True:
    newpid = os.fork()
    if newpid == 0:
        child()
        os._exit(0)
    else:   
        parent()
    childProcExitInfo = os.waitpid(outpid, os.WNOHANG)
    if childProcExitInfo[0] == outpid:
        break

cap.release()

# Closes all the frames
cv2.destroyAllWindows()
