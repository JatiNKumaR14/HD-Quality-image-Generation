import cv2
import glob
import os
import numpy as np
IMAGE_DIR = 'Dataset/'
os.mkdir("Resized")
i = 0
images_path = IMAGE_DIR 
print("Resizing")
for filename in glob.glob(images_path + '*.jpeg'):
    img = cv2.imread(filename)
    img = cv2.resize(img,(128,128))
    cv2.imwrite("Resized/image%04i.jpeg" %i, img)
    print("Done %d" %i)
    i = i + 1
=