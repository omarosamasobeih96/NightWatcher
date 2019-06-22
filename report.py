import os

out_path = "Anomaly-Detection/out"

folders = os.listdir(out_path)
for folder in folders:
    folder_path = os.path.join(out_path , folder)
    f = open(os.path.join(folder_path , "0_C.mat") , "r")
    lines = f.readlines()
    for line in lines:
        if float(line) >= 0.5:
            print("Error " + folder_path)
