
import numpy as np

def generate_xdata(x, data, index, win_size):
    for i in range(win_size, index):
        x.append(data.iloc[index[i - win_size:i, :]])
    return np.array(x)

def generate_ydata(y, data, index, win_size):
    for i in range(win_size, index):
        y.append(data['Net Chng'].iloc[index[i]])
    return np.array(y)