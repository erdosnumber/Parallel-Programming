import cv2
import numpy as np
import sys
import os

folder_path="./test"
files=os.listdir(folder_path)

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})
cnt = 0
filenames_and_values = []
for file in files:
    if(file.endswith(".png")):
        img = cv2.imread(folder_path+"/"+file,0)
        if img is None:
            cnt+=1
            print(file)
            continue
        if img.shape != [28,28]:
            img2 = cv2.resize(img,(28,28))
            
        img = img2.reshape(28,28,-1)

        #revert the image,and normalize it to 0-1 range
        img = img/255.0

        mat = np.matrix(img)
        actual_output = file[10:11]
        file = file[0:5]+".txt"

        filenames_and_values.append((file,actual_output))

        with open("pre-proc-img/"+file, 'w') as f:
            for row in mat.A:
                f.write(np.array2string(row, separator='')[1:-1])
                f.write("\n")


print(cnt)