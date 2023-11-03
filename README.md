# CNN Computer Vision - Two Models
This project consists of two computer vision models:

## Checking if a Person Has Sunglasses
## Emotion Recognition (Angry, Happy, Sad, Neutral)
# Model 1: Checking if a Person Has Sunglasses (has_sunglasses.py)
## Preprocessing
The code in has_sunglasses.py preprocesses images and converts PGM format to PNG format. It separates the training and testing datasets and labels images as having sunglasses (1) or not having sunglasses (0).

## Convolutional Neural Network (CNN)
The CNN model architecture is as follows:

Convolutional Layer with 32 filters, 3x3 kernel, and ReLU activation
Max-Pooling Layer (2x2)
Second Convolutional Layer with 32 filters, 3x3 kernel, and ReLU activation
Second Max-Pooling Layer (2x2)
Flattening Layer
Fully Connected Layer with 128 units and ReLU activation
Output Layer with 1 unit and Sigmoid activation
The CNN is compiled with the Adam optimizer and binary cross-entropy loss. It is trained on the training set and evaluated on the test set.

## Prediction
A test image is loaded, preprocessed, and passed through the trained model to predict whether the person in the image has sunglasses or not.

# Model 2: Emotion Recognition (emotion_rec.py)
## Preprocessing
The code in emotion_rec.py preprocesses images and converts PGM format to PNG format. It organizes the data into directories based on emotions (angry, happy, sad, neutral). The ImageDataGenerator is used for data augmentation.

## Convolutional Neural Network (CNN)
The CNN model architecture is as follows:

Convolutional Layer with 32 filters, 3x3 kernel, and ReLU activation
Max-Pooling Layer (2x2)
Second Convolutional Layer with 32 filters, 3x3 kernel, and ReLU activation
Second Max-Pooling Layer (2x2)
Flattening Layer
Fully Connected Layer with 128 units and ReLU activation
Output Layer with 4 units (for 4 emotions) and Softmax activation
The CNN is compiled with the Adam optimizer and categorical cross-entropy loss. It is trained on the training set and evaluated on the test set.

## Prediction
The code loops through the test images, loads each image, preprocesses it, and passes it through the trained model to predict the emotion of the person in the image (angry, happy, sad, or neutral).
