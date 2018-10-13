import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, LSTM
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
import pdb

class WindowSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, previous, future, sliding_window = True):

        self.future = future
        self.previous = previous
        self.sliding_window = sliding_window

    def transform(self, A):
        window = self.future + self.previous     #Number of samples per row (sample + target)

        if self.sliding_window:
            I = np.arange(window) + np.arange(A.shape[0] - window + 1).reshape(-1, 1)
        else:
            if A.shape[0] % self.future == 0:
                I = np.arange(window)+np.arange(0,A.shape[0]-window,self.future).reshape(-1,1)

            else:
                I = np.arange(window)+np.arange(0,A.shape[0] - window,window).reshape(-1,1)

        out = A[I].reshape(-1, window * A.shape[1], A.shape[2])
        ci = self.previous * A.shape[1]    #Number of features per sample
        return out[:, :ci], out[:, ci:] #Sample matrix, Target matrix

#data file path
dfp = 'bitcoin2015to2017_without_tweet.csv'

#Columns of price data to use
pred_columns = ['Close']

df = pd.read_csv(dfp)
timevals = df['Timestamp']
df = df.loc[:, pred_columns]
original_df = pd.read_csv(dfp).loc[:, pred_columns]

factor_scaling = MinMaxScaler()

# normalization
for c in pred_columns:
    df[c] = factor_scaling.fit_transform(df[c].values.reshape(-1,1))

#Features are input sample dimensions(channels)
values_to_be_predicted = np.array(df)[:,None,:]
timevals = np.array(timevals)[:,None,None]
original_values_to_be_predicted = np.array(original_df)[:,None,:]

#Make samples of temporal sequences of pricing data (channel)
past, fut = 256, 16
ps = WindowSampler(past, fut, sliding_window=False)

B, Y = ps.transform(values_to_be_predicted)
input_times, output_times = ps.transform(timevals)
original_B, original_Y = ps.transform(original_values_to_be_predicted)

datas = B
labels = Y
original_inputs = original_B
original_outputs = original_Y
output_file_name='bitcoin2015to2017_close_CNN_2_relu'

step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]
epochs = 100
units= 50
second_units = 30
output_size = 16

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]
#build model

# 4 layers
model = Sequential()
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
model.add(Dropout(0.5))
model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))
model.compile(loss='mse', optimizer='adam')
model.fit(training_datas, training_labels,verbose=1, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint('weights/'+output_file_name+'-{epoch:02d}.hdf5', monitor='val_loss', verbose=1,mode='min')])
original_datas = np.array(original_df)

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]

validation_original_outputs = original_outputs[training_size:,:,:]
validation_original_inputs = original_inputs[training_size:,:,:]
validation_input_times = input_times[training_size:,:,:]
validation_output_times = output_times[training_size:,:,:]
# pdb.set_trace()
ground_true = np.append(validation_original_inputs,validation_original_outputs, axis=1)
ground_true_times = np.append(validation_input_times,validation_output_times, axis=1)

step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]

#build model
# 4 layers
model = Sequential()
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
model.add(Dropout(0.25))
model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))
model.load_weights('weights/bitcoin2015to2017_close_CNN_2_relu-100.hdf5')
model.compile(loss='mse', optimizer='adam')
predicted = model.predict(validation_datas)
predicted_inverted = []

for i in range(original_datas.shape[1]):
    factor_scaling.fit(original_datas[:,i].reshape(-1,1))
    predicted_inverted.append(factor_scaling.inverse_transform(predicted[:,:,i]))

# pdb.set_trace()
#get only the close data
ground_true = ground_true[:,:,0].reshape(-1)
ground_true_times = ground_true_times.reshape(-1)
ground_true_times = pd.to_datetime(ground_true_times, unit='s')
# since we are appending in the first dimension
predicted_inverted = np.array(predicted_inverted)[0,:,:].reshape(-1)
validation_output_times = pd.to_datetime(validation_output_times.reshape(-1), unit='s')

ground_true_df = pd.DataFrame()
ground_true_df['times'] = ground_true_times
ground_true_df['value'] = ground_true

prediction_df = pd.DataFrame()
prediction_df['times'] = validation_output_times
prediction_df['value'] = predicted_inverted

plt.figure(figsize=(20,10))
plt.plot(ground_true_df['times'],ground_true_df['value'], 'bo', label = 'Actual')
plt.plot(prediction_df['times'],prediction_df['value'],'r-', label='Predicted')

with open("results.csv", "w") as f:
    print("inside")

    for i in range(len(prediction_df)):
        line =str(prediction_df.ix[i]['times']) + "," + str(prediction_df.ix[i]['value'])
        f.write(line+"\n")
#---------------------------------------------------------------------------------------------

#data file path
dfp = 'bitcoin2015to2017_with_tweetsentiment.csv'
#Columns of price data to use
pred_columns = ['Close']
df = pd.read_csv(dfp)
timevals = df.loc[:,['Timestamp', 'Sentiment Of Tweet']]
df = df.loc[:,pred_columns]
original_df = pd.read_csv(dfp).loc[:,pred_columns]

factor_scaling = MinMaxScaler()

# normalization
for c in pred_columns:
    df[c] = factor_scaling.fit_transform(df[c].values.reshape(-1,1))

#Features are input sample dimensions(channels)
values_to_be_predicted = np.array(df)[:,None,:]
timevals = np.array(timevals)[:,None,:]
original_values_to_be_predicted = np.array(original_df)[:,None,:]

#Make samples of temporal sequences of pricing data (channel)
past, fut = 256, 16
ps = WindowSampler(past, fut, sliding_window=False)
B, Y = ps.transform(values_to_be_predicted)
input_times, output_times = ps.transform(timevals)
original_B, original_Y = ps.transform(original_values_to_be_predicted)

sentiment_in = input_times
datas_input = B
datas = []
for i in range(B.shape[0]):
    list1 = []
    for j in range(B.shape[1]):
        list = []
        list.append(datas_input[i,j,0])
        list.append(sentiment_in[i,j,1])
        list1.append(list)
    datas.append(list1)

datas = np.array(datas)
labels = Y


output_file_name='bitcoin2015to2017_close_CNN_2_relu_tweet'
step_size = datas.shape[1]
nb_features = datas.shape[2]
batch_size= 8
epochs = 100
units= 50
second_units = 30
output_size = 16

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]
#build model

# 4 layers
model = Sequential()
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
model.add(Dropout(0.5))
model.add(Conv1D( strides=4, filters=1, kernel_size=16))
model.compile(loss='mse', optimizer='adam')
model.fit(training_datas, training_labels,verbose=1, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint('weights/'+output_file_name+'-{epoch:02d}.hdf5', monitor='val_loss', verbose=1,mode='min')])

original_inputs = original_B
original_outputs = original_Y
original_datas = np.array(original_df)

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]
validation_input_times = input_times[training_size:,:,:]
validation_output_times = output_times[training_size:,:,:]
validation_original_outputs = original_outputs[training_size:,:,:]
validation_original_inputs = original_inputs[training_size:,:,:]

step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]


#build model
# 4 layers
model = Sequential()
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
model.add(Dropout(0.25))
model.add(Conv1D( strides=4, filters=1, kernel_size=16))
model.load_weights('weights/bitcoin2015to2017_close_CNN_2_relu_tweet-100.hdf5')
model.compile(loss='mse', optimizer='adam')
predicted = model.predict(validation_datas)

predicted_inverted = []

for i in range(original_datas.shape[1]):
    factor_scaling.fit(original_datas[:,i].reshape(-1,1))
    predicted_inverted.append(factor_scaling.inverse_transform(predicted[:,:,i]))

ground_true = np.append(validation_original_inputs,validation_original_outputs, axis=1)
ground_true_times = np.append(validation_input_times[:,:,0],validation_output_times[:,:,1], axis=1)
#get only the close data
ground_true = ground_true[:,:,0].reshape(-1)
ground_true_times = ground_true_times.reshape(-1)
ground_true_times = pd.to_datetime(ground_true_times, unit='s')

# since we are appending in the first dimension
predicted_inverted = np.array(predicted_inverted)[0,:,:].reshape(-1)

ground_true_df = pd.DataFrame()
ground_true_df['times'] = ground_true_times
ground_true_df['value'] = ground_true

prediction_df = pd.DataFrame()
prediction_df['times'] = pd.to_datetime(validation_output_times[:, :, 0].reshape(-1), unit='s')
prediction_df['sentiment'] = validation_output_times[:, :, 1].reshape(-1)
prediction_df['value'] = predicted_inverted

plt.plot(prediction_df['times'],prediction_df['value'],'g-', label='Predicted_with_sentiment')
plt.legend(loc='upper left')
plt.title('CNN 2 Layers')
plt.xlabel('Timestamp')
plt.ylabel('Bitcoin Prices')
plt.savefig("CNN_2.png")

with open("results_tweet.csv", "w") as f:
    print("inside")

    for i in range(len(prediction_df)):
        line =str(prediction_df.ix[i]['times']) + "," + str(prediction_df.ix[i]['sentiment']) + "," + str(prediction_df.ix[i]['value'])
        f.write(line+"\n")

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

