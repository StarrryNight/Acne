import numpy as np
import pandas as pd
import os, sys 
from keras.applications import VGG16
from keras import models
import pickle
import helper as h
import time 
classifier_types = ['blackheads', 'whiteheads', 'nodules', 'dark spot', 'pustules']
X = []
y = []
import time
def warp_and_create_cnn_feature(image,modelvgg):
    '''
    image  : np.array of (N image, shape1, shape2, Nchannel )
    shape 1 and shape 2 depend on each image
    '''
    print("-"*10)
    print("warp_and_create_cnn_feature")
    start = time.time()
    print("len(image)={}".format(len(image)))
    print("**warp image**")
    for irow in range(len(image)):
        image[irow] = h.warp(image[irow],warped_size)
    image = np.array(image)
    print("**create CNN features**")
    feature = modelvgg.predict(image)
    print("DONE!")
    print("  feature.shape={}".format(feature.shape))
    end = time.time()
    print("  Time Took = {:5.2f}MIN".format((end - start)/60.0))
    print("")
    return(feature)

#nodules', 'dark spot', 'pustule
blackheads = None
whiteheads = None
nodules=None
dark_spot= None
pustules= None

modelvgg16 = VGG16(include_top=True,weights='imagenet')
modelvgg = models.Model(inputs  =  modelvgg16.inputs, 
                        outputs = modelvgg16.layers[-3].output)
warped_size = (224, 224)

name_to_var =  {
'blackheads':blackheads, 'whiteheads':whiteheads, 'nodules':nodules, 'dark spot':dark_spot, 'pustules':pustules
}
for idx, name in enumerate(classifier_types):
    with open(f'./result/{name}_positives.pickle', 'rb') as f:
        positives = pickle.load(f)
        print("Hi")
        print(positives)
    
    
        name_to_var[name] = (warp_and_create_cnn_feature(positives,modelvgg),idx)


## stack the sets of data
for key,value in name_to_var:
    X = np.concatenate(X,value[0])
    y = np.concatenate(y,value[0])


## Save data
dir_result = "result"
print("X.shape={}, y.shape={}".format(X.shape,y.shape))
np.save(file = os.path.join(dir_result,"X.npy"),arr = X)
np.save(file = os.path.join(dir_result,"y.npy"),arr = y)