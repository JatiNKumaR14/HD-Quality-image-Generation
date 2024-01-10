#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

import random
import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from zipfile import ZipFile
filepath="/content/drive/My Drive/Generated/dataset.zip"
with ZipFile(filepath,'r') as zip:
  zip.extractall()
print('Done')


# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# ## Explore the Data

# In[ ]:


covid_dir = os.path.join('dataset/train/covid')
normal_dir = os.path.join('dataset/train/normal')
pneumonia_dir = os.path.join('dataset/train/pneumonia')

print('Total training covid images:', len(os.listdir(covid_dir)))
print('Total training normal images:', len(os.listdir(normal_dir)))
print('Total training pneumonia images:', len(os.listdir(pneumonia_dir)))

covid_files = os.listdir(covid_dir)
print(covid_files[:10])

normal_files = os.listdir(normal_dir)
print(normal_files[:10])

pneumonia_files = os.listdir(pneumonia_dir)
print(pneumonia_files[:10])


# In[ ]:


pic_index = 2

next_covid = [os.path.join(covid_dir, fname) 
                for fname in covid_files[pic_index-2:pic_index]]
next_normal = [os.path.join(normal_dir, fname) 
                for fname in normal_files[pic_index-2:pic_index]]
next_pneumonia = [os.path.join(pneumonia_dir, fname) 
                for fname in pneumonia_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_covid+next_normal+next_pneumonia):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()


# ## Data Preprocessing

# In[ ]:


TRAINING_DIR = "dataset/train"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "dataset/val"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(224,224),
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(224,224),
    class_mode='categorical')


# ## Building Model

# In[ ]:


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ## Training

# In[ ]:


history = model.fit_generator(train_generator, epochs=60, validation_data = validation_generator, verbose = 1)

model.save("rps.h5")


# ## Evaluating Accuracy and Loss for the Model

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# **RUN THE BELOW 2 BLOCKS ONLY IF NECESSARY**

# In[53]:


def assignvalue(index):
  if(index==0):
    return "Covid"
  elif(index==1):
    return "Normal"
  else:
    return "Pneumonia"
def getval(output):
  result=np.where(output==1.0)
  lis=list(zip(result[0], result[1]))
  _,index=lis[0]
  return index
from keras.preprocessing import image
img1 = image.load_img("/content/covid1.jpg",target_size=(224,224))
img2 = image.load_img("/content/covid2.jpg",target_size=(224,224))
img3 = image.load_img("/content/Normal_xray.jpeg",target_size=(224,224))
img4 = image.load_img("/content/pneumonic.jpeg",target_size=(224,224))
img1 = np.asarray(img1)
img2 = np.asarray(img2)
img3 = np.asarray(img3)
img4 = np.asarray(img4)
img01 = np.expand_dims(img1, axis=0)
img02 = np.expand_dims(img2, axis=0)
img03 = np.expand_dims(img3, axis=0)
img04 = np.expand_dims(img4, axis=0)
saved_model=tf.keras.models.load_model("rps.h5")
output1 = saved_model.predict(img01)
output2 = saved_model.predict(img02)
output3 = saved_model.predict(img03)
output4 = saved_model.predict(img04)


# In[ ]:


f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4 , sharex=True , sharey=True)
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
ax4.imshow(img4)
#val1=output1+" : "+assignvalue(getval(output1))
ax1.set(xlabel=assignvalue(getval(output1))+": True")
ax2.set(xlabel=assignvalue(getval(output2))+": True")
ax3.set(xlabel=assignvalue(getval(output3))+": True")
ax4.set(xlabel=assignvalue(getval(output4))+": False")


# In[ ]:


# load all images into a list
images = []
img_folder = os.path.join('test')
img_files = os.listdir(img_folder)
img_files = [os.path.join(img_folder, f) for f in img_files]
img_files.remove("test/.ipynb_checkpoints")
print(img_files)
for img in img_files:
    img = load_img(img, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
# print(images)
classes = model.predict_classes(images, batch_size=10)
def assignvalue(index):
  if(index==0):
    return "Covid"
  elif(index==1):
    return "Normal"
  else:
    return "Pneumonia"
for i in range(len(img_files)):
  print(img_files[i]," ---- ",assignvalue(classes[i]))


# ## Visualizing Intermediate Representations

# In[ ]:


# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]

#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Let's prepare a random input image of a covid,normal or pneumonia from the training set.
covid_img_files = [os.path.join(covid_dir, f) for f in covid_files]
normal_img_files = [os.path.join(normal_dir, f) for f in normal_files]
pneumonia_img_files = [os.path.join(pneumonia_dir, f) for f in pneumonia_files]

img_path = random.choice(covid_img_files + normal_img_files+pneumonia_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
        x  = feature_map[0, :, :, i]
        x -= x.mean()
        x /= x.std ()
        x *=  64
        x += 128
        x  = np.clip(x, 0, 255).astype('uint8')
        display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )


# In[ ]:




