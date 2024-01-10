import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import random
import os


def assignvalue(index):
  if(index==0):
    return "Covid"
  elif(index==1):
    return "Normal"
  else:
    return "Pneumonia"
from keras.preprocessing import image
img = image.load_img("/content/IMG-20200616-WA0056.jpg",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

saved_model=tf.keras.models.load_model("rps.h5")
output = saved_model.predict(img)
result=np.where(output==1.0)
lis=list(zip(result[0], result[1]))
_,index=lis[0]
print (assignvalue(index))