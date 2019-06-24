import tkinter as tk
import os
import signal

is_running = False

def start():
    global is_running
    if is_running == 0:
        os.kill(os.getppid(), signal.SIGALRM)
    is_running = True

def stop():
    if is_running == 1:
        os.kill(os.getppid(), signal.SIGALRM)
        quit()

def run():
    root = tk.Tk()
    root.title("Night Watcher")
    logo = tk.PhotoImage(file="resources/logo.png")
    start_button = tk.PhotoImage(file="resources/start.png")
    stop_button = tk.PhotoImage(file="resources/stop.png")
    tk.Label(root, image=logo).pack()
    frame = tk.Frame(root)
    frame.pack()

    button = tk.Button(frame, 
                    image=start_button,width=350,height=150,
                    command=start)
    button.pack(side=tk.LEFT)

    slogan = tk.Button(frame,
                    image=stop_button,width=350,height=150,
                    command=stop)
    slogan.pack(side=tk.LEFT)

    root.mainloop()