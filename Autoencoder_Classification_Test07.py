import os
import pandas as pd 
import tensorflow as tf
import numpy as np 
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt 
import scipy.io as scio
from numpy.random import seed
import time
import keras
from keras import callbacks
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Masking
from keras.layers import Concatenate
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.layers import Flatten
from keras.layers import Reshape
from keras import losses
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
# Read the data
elapse0 = time.time()  # start timing
# orbits = pd.read_csv('D:/ops/results/paper31-STELLA_Eglin-1sigma_1sigma_true_orbit_impulseManeuver.csv')
train_input_1 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/training_input_1_data.mat')
train_input_1 = train_input_1['training_input_1_data']
train_input_2 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/training_input_2_data.mat')
train_input_2 = train_input_2['training_input_2_data']
train_output = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/training_output_data.mat')
train_output = train_output['training_output_data']

test_1_input_1 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_1_input_1_data.mat')
test_1_input_1 = test_1_input_1['testing_1_input_1_data']
test_1_input_2 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_1_input_2_data.mat')
test_1_input_2 = test_1_input_2['testing_1_input_2_data']
test_1_output = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_1_output_data.mat')
test_1_output = test_1_output['testing_1_output_data']

# validation_1 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/validation_1.mat')
# validation_1 = validation_1['validation_1']
# validation_2 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/validation_2.mat')
# validation_2 = validation_2['validation_2']

test_2_input_1 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_2_input_1_data.mat')
test_2_input_1 = test_2_input_1['testing_2_input_1_data']
test_2_input_2 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_2_input_2_data.mat')
test_2_input_2 = test_2_input_2['testing_2_input_2_data']
test_2_output = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_2_output_data.mat')
test_2_output = test_2_output['testing_2_output_data']

test_3_input_1 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_3_input_1_data.mat')
test_3_input_1 = test_3_input_1['testing_3_input_1_data']
test_3_input_2 = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_3_input_2_data.mat')
test_3_input_2 = test_3_input_2['testing_3_input_2_data']
test_3_output = scio.loadmat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/testing_3_output_data.mat')
test_3_output = test_3_output['testing_3_output_data']

# fix random seed for reproducbility
# np.random.seed(7)
np.random.seed(12)

# define the input

# define the input
timesteps = train_input_1.shape[1]
n_features = train_input_1.shape[2]
faltten_dim = timesteps * n_features
activation_function_1 = 'relu'
activation_function_2 = 'relu'
activation_function_3 = 'tanh'
num_neuron_1 = 12
num_neuron_2 = 6
# num_neuron_3 = 6
# num_neuron_4 = 8
# define the model


input_1 = Input(shape=(timesteps, n_features))
faltten_1 = Flatten()(input_1)
encoder_1 = Dense(num_neuron_1, activation=activation_function_1)(faltten_1)
encoder_1 = Dense(num_neuron_2, activation=activation_function_2)(encoder_1)
# encoder_1 = Dense(num_neuron_3, activation=activation_function_2)(encoder_1)
# encoder_1 = Dense(num_neuron_4, activation=activation_function_2)(encoder_1)

# input 2 autoencoder layer
input_2 = Input(shape=(timesteps, n_features))
faltten_2 = Flatten()(input_2)
encoder_2 = Dense(num_neuron_1, activation=activation_function_1)(faltten_2)
encoder_2 = Dense(num_neuron_2, activation=activation_function_2)(encoder_2)
# encoder_2 = Dense(num_neuron_3, activation=activation_function_2)(encoder_2)
# encoder_2 = Dense(num_neuron_4, activation=activation_function_2)(encoder_2)

# concatenate encoding layers
c_encoded = Concatenate(name="concat", axis=1)([encoder_1, encoder_2])
encodered = Dense(6, activation=activation_function_2)(c_encoded)
encodered = Dense(4, activation=activation_function_3)(encodered)
# encodered = Dense(3, activation=activation_function)(encodered)
feature = Dense(1, activation = 'sigmoid')(encodered)

# Now we have two input and two output with shared layer  
model = Model([input_1, input_2], feature) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary() 





model.summary()

# plot_model(model, to_file='D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/Network_LayerwithFlatten.png', show_shapes=True, show_layer_names=True)
num_epoch =800
batchsize = 128 #train.shape[0]
validationsplit = 0.2  # 30%
verboseshow = 0

# fit model

early_stopping=EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min', restore_best_weights=True)
history = model.fit([train_input_1, train_input_2],train_output, epochs=num_epoch, batch_size=batchsize, 
                    verbose=verboseshow, validation_split = validationsplit, callbacks=[early_stopping])


# Autoencode = Model([input_1, input_2], encodered)
# plot_model(Autoencode, to_file='D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/autoencoder_withFlatten.png', show_shapes=True, show_layer_names=True)

# save the model
# model.save('TensorLSTM_test02.h5')
print(f'# trained in {time.time()-elapse0:.2f}s.')

# make prediction
train_pred = model.predict([train_input_1, train_input_1])
test_1_prediction = model.predict([test_1_input_1, test_1_input_2])
test_2_prediction = model.predict([test_2_input_1, test_2_input_2])
test_3_prediction = model.predict([test_3_input_1, test_3_input_2])

plt.plot(history.history['loss'], linewidth=2, label='Train')
plt.plot(history.history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.savefig('D:/ops/result_3/results-20200317/sample_down_data_collection/TwoTrackIndependent/fig/model-loss.png')


scio.savemat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/train_pred.mat',{'train_pred':train_pred})
scio.savemat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/test_1_prediction.mat',{'test_1_prediction':test_1_prediction})
scio.savemat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/test_2_prediction.mat',{'test_2_prediction':test_2_prediction})
scio.savemat('D:/ops/GPclassification/CollectDatabase/AutoEncoder/TwoIndependent/AutoencoderResult/test_3_prediction.mat',{'test_3_prediction':test_3_prediction})
print(n_features)
plt.show()