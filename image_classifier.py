from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import cv2
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 150, 150

#train_data_dir = '/content/drive/My Drive/Colab Notebooks/ice_project/data/train'
#validation_data_dir = '/content/drive/My Drive/Colab Notebooks/ice_project/data/validation'
#nb_train_samples = 769
#nb_validation_samples = 250
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = load_model('model.h5')

#img = cv2.imread('/content/drive/My Drive/Colab Notebooks/ice_project/data/train/day/frame1569.jpg')
img = cv2.imread('test_images/test11.jpg')

np_image = np.array(img).astype('float32')/255
np_image = transform.resize(np_image, (150, 150, 3))
cv2.imshow('transformed', np_image)
np_image = np.expand_dims(np_image, axis=0)

result= model.predict(np_image)

if result[0][0] > 0.5:
  label = 'night'
if result[0][0] <= 0.5:
  label = 'day'
  
print(label)
cv2.putText(img, label , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, bottomLeftOrigin=False)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()