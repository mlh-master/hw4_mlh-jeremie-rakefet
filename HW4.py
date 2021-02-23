# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# + pycharm={"is_executing": false}
import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# + pycharm={"is_executing": false}
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# + pycharm={"is_executing": false}
def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# -

def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)

keras.backend.clear_session()

# ### PART 1: Fully connected layers 
# --------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *NN with fully connected layers. 
#
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
model_relu = Sequential()
model_relu.add(Flatten(input_shape=(32,32,1)))
model_relu.add(Dense(300, activation="relu", kernel_initializer='he_normal'))
model_relu.add(Dense(150, activation="relu", kernel_initializer='he_normal'))
model_relu.add(Dense(4, activation='softmax', kernel_initializer='he_normal'))
#----------------------------------------------------------
#
# ------------------------------
# -
model_relu.summary()

# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

# -

# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task. Train your model on the training set and evaluate the model on the testing set. Print the accuracy and loss over the testing set.
# +
#--------------------------Impelment your code here:-------------------------------------
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
model_relu.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs= epochs)
print("Loss and accuracy of model_relu batch_size 64 epochs 25:")
loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)

#----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 2:***</span> *Activation functions.* 
#
# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model = Sequential()
new_a_model.add(Flatten(input_shape=(32,32,1)))
new_a_model.add(Dense(300, activation="sigmoid", kernel_initializer='he_normal'))
new_a_model.add(Dense(150, activation="sigmoid", kernel_initializer='he_normal'))
new_a_model.add(Dense(4, activation='softmax', kernel_initializer='he_normal'))
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
#----------------------------------------------------------------------------------------
# -

new_a_model.summary()

# ---
# <span style="color:red">***Task 3:***</span> *Number of epochs.* 
#
# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*
#
# ---

# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model.fit(BaseX_train, BaseY_train,batch_size=batch_size, epochs= epochs)
print("Loss and accuracy of new_a_model with epoch 25 batch_size 64:")
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)

if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "new_a_model_epoches25_weights.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)
#-----------------------------------------------------------------------------------------

# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)



# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model.fit(BaseX_train, BaseY_train,batch_size=batch_size, epochs= epochs)
print("Loss and accuracy of new_a_model with epoch 40 batch size 64:")
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)

if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "new_a_model_epoches40_weights.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)
#-----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 4:***</span> *Mini-batches.* 
#
# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*
#
# ---

keras.backend.clear_session()

# +
#--------------------------Impelment your code here:-------------------------------------
model_relu = Sequential()
model_relu.add(Flatten(input_shape=(32,32,1)))
model_relu.add(Dense(300, activation="relu"))
model_relu.add(Dense(150, activation="relu", kernel_initializer='he_normal'))
model_relu.add(Dense(4, activation='softmax', kernel_initializer='he_normal'))
#----------------------------------------------------------------------------------------

# +
batch_size = 32
epochs = 50


#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# +
#--------------------------Impelment your code here:-------------------------------------
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
model_relu.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs= epochs)
print("Loss and accuracy of model_relu with epochs 50 and batch size of 32:")
loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)
#----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 4:***</span> *Batch normalization.* 
#
# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*
#
# ---

keras.backend.clear_session()

# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model = Sequential()
new_a_model.add(Flatten(input_shape=(32,32,1)))
new_a_model.add(Dense(300, activation="sigmoid", kernel_initializer='he_normal'))
new_a_model.add(BatchNormalization())
new_a_model.add(Dense(150, activation="sigmoid", kernel_initializer='he_normal'))
new_a_model.add(BatchNormalization())
new_a_model.add(Dense(4, activation='softmax', kernel_initializer='he_normal'))
#---------------------------------------------------------------------------------------

# +
batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network: 


# +
#Preforming the training by using fit 
#--------------------------Impelment your code here:-------------------------------------
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
new_a_model.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs= epochs)
print("Loss and accuracy of new_a_model with batch_size 32 batch epochs 50 normalization:")
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)
#----------------------------------------------------------------------------------------
# -

# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.* 
#
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
#
# ---

def get_net(input_shape,drop,dropRate,reg):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:    
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:      
    model.add(Dense(512, activation='elu',name='FCN_1')) 
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model


input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
NNet = get_net(input_shape,drop,dropRate,reg)

# + pycharm={"is_executing": false}
NNet=get_net(input_shape,drop,dropRate,reg)

# + pycharm={"is_executing": false}
from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
# -

#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)

# + pycharm={"is_executing": false}
# NNet.load_weights('Weights_1.h5')

# + pycharm={"is_executing": false}
results = NNet.evaluate(X_test,Y_test)
print('test loss, test acc:', results)
# -

# ---
# <span style="color:red">***Task 2:***</span> *Number of filters* 
#
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
def get_net(input_shape,drop,dropRate,reg, filters):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Conv2D(filters=filters[1], kernel_size=(3, 3), padding='same', activation='relu', name='Conv2D_1',
                     kernel_regularizer=regularizers.l2(reg)))
    for filter in filters[2:]:
        if drop:
            model.add(Dropout(rate=dropRate))
        model.add(BatchNormalization(axis=1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=filter, kernel_size=(3,3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:
    model.add(Dense(512, activation='elu',name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model
#----------------------------------------------------------------------------------------
# -
filters = [64, 128, 128, 256, 256]
NNet2 =get_net(input_shape,drop,dropRate,reg, filters)
NNet2.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')
h2 = NNet2.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
results = NNet2.evaluate(X_test,Y_test)
print('Part2 network with filters: [64, 128, 128, 256, 256], test loss, test acc:', results)

filters = [int(x / 2) for x in filters]
NNet2_half =get_net(input_shape,drop,dropRate,reg, filters)
NNet2_half.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')
h2_half = NNet2_half.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
results = NNet2_half.evaluate(X_test,Y_test)
print('Part2 network with filters: [32, 64, 64, 128, 128],test loss, test acc:', results)
# That's all folks! See you :)
