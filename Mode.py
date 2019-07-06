import tkinter as tk
from tkinter import filedialog as fd
import os
import signal


def isr(signum, frame):
    quit()


def video():
    fname = fd.askopenfilename(title = "Select Video",filetypes=(("Video files", "*.mp4"),
                                       ("All files", "*.*") ))
    f = open("videopath.txt", "w")
    f.write(fname)
    f.close()
    print(str(fname))
    os.kill(os.getppid(), signal.SIGALRM)
    quit()

def camera():
    quit()

def run():
    root = tk.Tk()
    signal.signal(signal.SIGALRM, isr)
    root.title("Night Watcher")
    logo = tk.PhotoImage(file="resources/logo.png")
    start_button = tk.PhotoImage(file="resources/video.png")
    stop_button = tk.PhotoImage(file="resources/camera.png")
    tk.Label(root, image=logo).pack()
    frame = tk.Frame(root)
    frame.pack()

    button = tk.Button(frame, 
                    image=start_button,width=350,height=150,
                    command= video)
    button.pack(side=tk.LEFT)

    slogan = tk.Button(frame,
                    image=stop_button,width=350,height=150,
                    command=camera)
    slogan.pack(side=tk.LEFT)

    root.mainloop()