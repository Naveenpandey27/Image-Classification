# import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training set
# define ImageDataGenerator object for training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# load training set
training_set = train_datagen.flow_from_directory('basedata/training_set',
                                                 target_size = (200, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
# define ImageDataGenerator object for test set
test_datagen = ImageDataGenerator(rescale = 1./255)
# load test set
test_set = test_datagen.flow_from_directory('basedata/validation_set',
                                            target_size = (200, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Initialising the CNN
# define CNN model as Sequential
cnn = tf.keras.models.Sequential()

# Convolution
# add 1st convolutional layer to the CNN model
cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[200, 200, 3]))

# Pooling
# add max pooling layer to the CNN model
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
# add 2nd convolutional layer to the CNN model
cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
# flatten the CNN model
cnn.add(tf.keras.layers.Flatten())

# Full Connection
# add dense layer to the CNN model
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
## For Binary Classification
# add output layer for binary classification with l2 regularization
cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation ='linear'))

# print model summary
cnn.summary()

# Compiling the CNN
# compile the CNN model with optimizer, loss, and evaluation metric
cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
# train the CNN model on training set and evaluate it on test set
r=cnn.fit(x = training_set, validation_data = test_set, epochs = 50)

# save the CNN model
cnn.save('model_SVM.h5')

# load the saved CNN model
from tensorflow.keras.models import load_model
model = load_model('model_SVM.h5')

# load and predict the class of an image using the loaded model
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/navee/Downloads/computer_vision/basedata/test_set/NF_S3_B3_35.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]>0:
    print('feed is not available')
else:
    print('feed is available')


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/navee/Downloads/computer_vision/basedata/test_set/NF_S4_B1_161.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]>0:
    print('feed is not available')
else:
    print('feed is available')

    
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/navee/Downloads/computer_vision/basedata/test_set/NF_S4_B2_26.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]>0:
    print('feed is not available')
else:
    print('feed is available')


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/navee/Downloads/computer_vision/basedata/test_set/NF_S6_B2_22.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]>0:
    print('feed is not available')
else:
    print('feed is available')


import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('C:/Users/navee/Downloads/computer_vision/basedata/test_set/NF_S6_B2_23.jpg', target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
if result[0]>0:
    print('feed is not available')
else:
    print('feed is available')


loaded_model = tf.keras.models.load_model('model_SVM.h5')
loaded_model.layers[0].input_shape #(None, 200, 200, 3)


# Testing on videos
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

dir_path = r'C:\Users\navee\Downloads\computer_vision\basedata\test_set/'      

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+ '//' + i, target_size = (200, 200))
    plt.imshow(img)
    plt.show()
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis = 0)
    result = loaded_model.predict(img)
    if result[0] > 0:
        print('feed is avilable')
    else:
        print('feed is available')
