# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:21:45 2023

@author: adina
"""

import os 
import pandas as pd 
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
from keras.preprocessing import image


def convert_pgm_to_png(input_dir):
    data = []
    
    for directory in input_dir:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".pgm"):
                    pgm_path = os.path.join(root, file)
                    img = Image.open(pgm_path)
                    png_filename = os.path.splitext(file)[0] + ".png"
                    png_path = os.path.join(root, png_filename)
                    img.save(png_path, "PNG")
                    
                    if "sunglasses" in file.lower():
                        has_sunglasses = '1'  # Image has sunglasses
                    else:
                        has_sunglasses = '0' 
                    
                    data.append({
                        'Filename': png_filename,
                        'Path': png_path,
                        'HasSunglasses': has_sunglasses
                    })

    df = pd.DataFrame(data)
    return df
    return 

# preproccesing the train set

directories_for_train = ['faces/an2i' , 'faces/at33' , 'faces/boland' , 'faces/bpm']

dataset_train = convert_pgm_to_png(directories_for_train)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


training_set = train_datagen.flow_from_dataframe(
    dataframe=dataset_train,
    x_col='Path',
    y_col='HasSunglasses',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


# preproccesing the test set
directories_for_test = ['faces/ch4f' , 'faces/cheyer' , 'faces/choon' , 'faces/danieln']

dataset_test = convert_pgm_to_png(directories_for_test)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


test_set = test_datagen.flow_from_dataframe(
    dataframe=dataset_test,
    x_col='Path',
    y_col='HasSunglasses',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


# intializing the cnn 
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)


# prediction
test_image = image.load_img('image_67184129.JPG', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'have sunglasses!'
else:
    prediction = 'not have sunglasses!'
print(prediction)