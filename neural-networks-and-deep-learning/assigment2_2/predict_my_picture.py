#coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr import predict, num_px, classes

# 这个样例中的图片尺寸为 64

f = open('lr.model', 'rb')
lr_model = pickle.load(f)

my_image = "./my_image.jpg"   # change this to the name of your image file

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(lr_model["w"], lr_model["b"], my_image)
plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")