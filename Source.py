import os
import time
import cv2
import numpy as np
import pipeline

# Constants
FPS = 16

# Remove previous videos
os.system("rm videos/*")

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
frames_cnt = 0
newpid = 0

# Prediction
def child():
    if frames_cnt == 0:
	    time.sleep(5)
    if frames_cnt == 0:
	    os._exit(0)

    pipeline.pipeline(path, frames_cnt)

    os._exit(0)

# Read from Camera and Store
id = 1
def parent():
    global frames_cnt, path, id
    frames_cnt = 0

    path = "video" + str(id) + ".mp4"

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter("videos/" + path,cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width,frame_height))
    

    while True:
        ret, frame = cap.read()
        
        # Write the frame
        if ret == True: 
            out.write(frame)
            frames_cnt = frames_cnt + 1
        else:
            os._exit(0)
            
        childProcExitInfo = os.waitpid(newpid, os.WNOHANG)
        if childProcExitInfo[0] == newpid:
            break


    id += 1
    out.release()

while True:
    newpid = os.fork()
    if newpid == 0:
        child()
    else:   
        parent()

cap.release()

# Closes all the frames
cv2.destroyAllWindows()
