#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
data_dir = pathlib.Path('DATASET') #Path of Photos
image_count = len(list(data_dir.glob('*/*.jpeg')))
print(image_count)
# rocketlaunches = list(data_dir.glob('rocketlaunches/*'))
# im = Image.open(str(rocketlaunches[0]))
batch_size = 32
img_height = 100
img_width = 100
data_train = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2,subset="training",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
print("Your Training Data : ",  data_train)
data_test = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.2, subset="validation",seed=123, image_size=(img_height, img_width),batch_size=batch_size)
class_names = data_train.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in data_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
print(model.summary())
epochs=10
Image_Model = model.fit(data_train,validation_data=data_test,epochs=epochs)
print('Accuracy Status : ',Image_Model.history['accuracy'])

#testing the model

img = tf.keras.utils.load_img('check.jpg', target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
result = model.predict(img_array)
print(result)
print(f"This is image of {class_names[np.argmax(result)]}")
# %%
