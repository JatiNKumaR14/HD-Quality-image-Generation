import numpy as np 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import random
import cv2
import os
import glob

i = 1

IMAGE_DIR = 'NORMAL/'

images_path = IMAGE_DIR 
print("Augmenting")

for img_name in glob.glob(images_path + '*.jpeg'):
    img = cv2.imread(img_name)
    img_flip_lr = cv2.flip(img, 1)
    cv2.imwrite('Augmented/Augmented_%d.jpg'%i, img_flip_lr)
    print("Done %d" %i)
    i = i + 1