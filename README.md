# Image-Classification


This Python code is a Convolutional Neural Network (CNN) model designed to classify images as 'feed available' or 'feed not available'.

Requirements
This code was written using Python 3.8.10 and the following packages:

TensorFlow 2.6.0
Keras 2.6.0
NumPy 1.19.5
Dataset
The model is trained on two sets of images, the training_set and the validation_set. Each set is divided into two subdirectories, 'feed_available' and 'feed_not_available'. The model is trained to classify images into one of these two categories.

Preprocessing
The training and validation sets are preprocessed using the ImageDataGenerator class of TensorFlow's preprocessing.image module. The training set is rescaled, sheared, zoomed, and flipped horizontally, while the validation set is only rescaled.

Model Architecture
The CNN model consists of the following layers:

Conv2D layer with 32 filters, kernel size of 3x3, stride of 2, and ReLU activation function
MaxPooling2D layer with pool size of 2x2 and stride of 2
Conv2D layer with 32 filters, kernel size of 3x3, and ReLU activation function
MaxPooling2D layer with pool size of 2x2 and stride of 2
Flatten layer
Dense layer with 128 units and ReLU activation function
Output layer with a single unit, a linear activation function, and L2 regularization
The model is compiled with the Adam optimizer, hinge loss function, and accuracy metric.

Training and Evaluation
The model is trained on the training set with 50 epochs and a batch size of 32. The model is then evaluated on the validation set.

Saving and Loading the Model
The trained model is saved to a file named model_SVM.h5 using Keras' save function. To load the model, the load_model function is used.

Testing
The model is tested on four test images. Each test image is loaded and preprocessed using TensorFlow's preprocessing.image module. The model then classifies the image as 'feed available' or 'feed not available' and outputs the result.
