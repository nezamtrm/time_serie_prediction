
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from gen_data import *

#preparing data
data = pd.read_csv('watchlist.csv')

data = data.drop('symbol', axis=1)
win_size = 20

train_index = 0.8*data.shape[0]
test_index = 0.2*data.shape[0]

xtrain, ytrain, xtest, ytest =[], [], [], []

xtrain = generate_xdata(xtrain, data, train_index, win_size)
xtest = generate_xdata(xtest, data, test_index, win_size)

ytrain = generate_ydata(ytrain, data, train_index, win_size)
ytest = generate_ydata(ytest, data, test_index, win_size)

#generating model
model = Sequential()
model.add(LSTM(50, activation='relu'), input_shape=(win_size, data.shape[1]))
model.add(Dense(1))
all_scores, all_mse = [], []
checkpoint = keras.callbacks.ModelCheckpoint("model.h5", monitor='val_accuracy', save_freq="epoch", save_best_only=True)

model.compile(optimizer='Adamax', loss='mean_squared_error', metrics=['accuracy', ])

history1 = model.fit(xtrain, ytrain, validation_batch_size=512, epochs=100,
                     validation_data=(xtest, ytest), batch_size=2048,
                     verbose=2, callbacks=[checkpoint])

val_mse, val_mae = model.evaluate(xtrain, ytrain, verbose=2)
all_scores.append(val_mae)
all_mse.append(val_mse)

print(f'val_mse:{val_mse}, val_mae:{val_mae}')

output = model.predict(xtest)

plt.plot(output, label='Predicted')
plt.plot(ytest, label='Real')

plt.xlabel('minute')
plt.ylabel('price')
plt.legend()
plt.show()
