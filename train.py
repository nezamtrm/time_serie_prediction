
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from gen_data import *

#preparing data
data = pd.read_csv('watchlist.csv')

data = data.drop('Symbol', axis=1)
data_x = data.drop('Net Chng', axis=1)
data_y = data['Net Chng']
win_size = 20

train_index = round(0.8*data.shape[0])
test_index = round(0.2*data.shape[0])

data_train = generate_data(data_x, data_y, train_index, win_size)
data_test = generate_data(data_x, data_y, test_index, win_size)
# x_train = generate_xdata(data_x, train_index, win_size)
# x_test = generate_xdata(data_x, train_index, win_size)
# y_train = generate_ydata(data_x, test_index, win_size)
# y_test = generate_ydata(data_x, test_index, win_size)

input_shape = (win_size, data_x.shape[1])

#generating model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=input_shape))
model.add(Dense(1, activation='sigmoid'))
all_scores, all_mse = [], []
checkpoint = keras.callbacks.ModelCheckpoint("model.h5", monitor='val_accuracy', save_freq="epoch", save_best_only=True)

model.compile(optimizer='Adamax', loss='mean_squared_error', metrics=['accuracy', ],)
# print(data_train)
history1 = model.fit(data_train[0], data_train[1], validation_batch_size=512, epochs=10,
                     validation_data=data_test, batch_size=2048,
                     verbose=2, callbacks=[checkpoint])

val_mse, val_mae = model.evaluate(data_train[0], data_train[1], verbose=2)
all_scores.append(val_mae)
all_mse.append(val_mse)

print(f'val_mse:{val_mse}, val_mae:{val_mae}')

output = model.predict(data_test[0])

plt.plot(output, label='Predicted')
plt.plot(data_test[1], label='Real')

plt.xlabel('minute')
plt.ylabel('price')
plt.legend()
plt.show()
