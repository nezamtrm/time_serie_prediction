import pandas as pd
from sklearn import preprocessing
import numpy as np

#The following function generates windows of input data and corresponding output data for a given window size and index position:
def generate_data(data_x, data_y, index, win_size):
    x, y = [], []
    for i in range(win_size, index):
        x.append(data_x.iloc[i-win_size:i,:])
        y.append(data_y.iloc[i])
    return np.array(x), np.array(y)

#The following function scales input data using StandardScaler from scikit-learn library and returns both the scaler and the scaled data:
def preprcss_data(data):
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data) #returns np array
    return scaler, pd.DataFrame(data_scaled)

