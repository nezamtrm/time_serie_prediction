
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from gen_data import *

#preparing data
data = pd.read_csv('data1.csv')

cols = ['Open', 'Low', 'High', 'Close']
# scaler, data[cols] = preprcss_data(data[cols]) #to preserve the column indexes
data = data[cols]

data_x = data.drop('Close', axis=1)
data_y = data['Close']
win_size = 20

train_index = round(0.8*data.shape[0])
test_index = round(0.2*data.shape[0])

data_train = generate_data(data_x, data_y, train_index, win_size)
data_test = generate_data(data_x, data_y, test_index, win_size)

input_shape = (win_size, data_x.shape[1])

#generating model
model = Sequential()
model.add(LSTM(150,  input_shape=input_shape, return_sequences=True))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(50,))
model.add(Dense(1,))
all_scores, all_mse = [], []


#Creating a callback to save the model weights
checkpoint = keras.callbacks.ModelCheckpoint("model_100epoch.h5", monitor='val_accuracy', save_freq="epoch", save_best_only=True)

model.compile(optimizer='Adamax', loss='mean_squared_error', metrics=['accuracy', ],)

#Training the model and monitoring the validation metrics
history1 = model.fit(data_train[0], data_train[1], validation_batch_size=32, epochs=100,
                     validation_data=data_test, batch_size=32,
                     verbose=2, callbacks=[checkpoint])
#Evaluating the model on validation data
val_mse, val_mae = model.evaluate(data_train[0], data_train[1], verbose=2)
all_scores.append(val_mae)
all_mse.append(val_mse)

print(f'val_mse:{val_mse}, val_mae:{val_mae}')

output = model.predict(data_test[0])
np.save('output.npy', output)

# output = scaler.inverse_transform(output)
# data_test[1] = scaler.inverse_transform(data_test[1])


plt.plot(output, label='Predicted')
plt.plot(data_test[1], label='Real')

plt.xlabel('minute')
plt.ylabel('price')
plt.legend()
plt.show()
