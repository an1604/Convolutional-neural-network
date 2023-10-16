import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import tensorflow as tf

# Function to convert PGM images to PNG format
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

                    if "angry" in file.lower():
                        emotion = 'angry'
                    elif "sad" in file.lower():
                        emotion = 'sad'
                    elif "happy" in file.lower():
                        emotion = 'happy'
                    else:
                        emotion = 'regular'
                    data.append({
                        'Filename': png_filename,
                        'Path': png_path,
                        'Emotion': emotion
                    })

    df = pd.DataFrame(data)
    return df

# Function to convert PGM images to PNG format
# (rest of your code remains the same)

# Directories for training and testing
directories_for_train = ['faces/an2i', 'faces/at33', 'faces/boland', 'faces/bpm']
directories_for_test = ['faces/ch4f', 'faces/cheyer', 'faces/choon', 'faces/danieln']

# Convert PGM to PNG and create datasets
dataset_train = convert_pgm_to_png(directories_for_train)
dataset_test = convert_pgm_to_png(directories_for_test)

# Organize data into directories based on emotions
for emotion in ['angry', 'happy', 'sad', 'regular']:
    emotion_data_train = dataset_train[dataset_train['Emotion'] == emotion]
    emotion_data_test = dataset_test[dataset_test['Emotion'] == emotion]

    # Create directories if they don't exist
    train_dir = f'train/{emotion}'
    test_dir = f'test/{emotion}'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move files to appropriate directories
    for index, row in emotion_data_train.iterrows():
        os.rename(row['Path'], os.path.join(train_dir, row['Filename']))

    for index, row in emotion_data_test.iterrows():
        os.rename(row['Path'], os.path.join(test_dir, row['Filename']))

# Use ImageDataGenerator.flow_from_directory
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    'train',  # Directory containing training data
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'test',  # Directory containing training data
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


# Initialize the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=100)

# prediction
import os
from keras.preprocessing import image
import numpy as np

# Define the directory containing the test images
test_directory = 'test'

# Get a list of subdirectories (each subdirectory corresponds to an emotion)
emotion_subdirectories = [os.path.join(test_directory, subdir) for subdir in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory, subdir))]

predictions = []
emotion_labels = ['angry', 'happy', 'sad', 'neutral']

for subdir in emotion_subdirectories:
    # Get a list of image files in the current emotion subdirectory
    image_files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in image_files:
        test_image = image.load_img(image_path, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        predicted_emotion = emotion_labels[np.argmax(result)]
        predictions.append((image_path, predicted_emotion))

# Display predictions for each image
for image_path, predicted_emotion in predictions:
    print(f"Image: {image_path}, Predicted Emotion: {predicted_emotion}")

