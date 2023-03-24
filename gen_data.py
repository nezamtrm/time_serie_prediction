
import numpy as np

# def generate_data(data_x, data_y, indx, win_size):
#     x, y = [], []
#     for i in range(win_size, indx):
#         x.append(data_x.iloc[i - win_size:i, :])
#         y.append(data_y.iloc[i])
#         return np.array(x), np.array(y)


# def generate_xdata(data_x, index, win_size):
#     x = []
#     for i in range(win_size, index):
#         x.append(data_x.iloc[i-win_size:i,:])
#     return np.array(x)
# def generate_ydata(data_y, index, win_size):
#     y = []
#     for i in range(win_size, index):
#         y.append(data_y.iloc[i])
#     return np.array(y)
def generate_data(data_x, data_y, index, win_size):
    x, y = [], []
    for i in range(win_size, index):
        x.append(data_x.iloc[i-win_size:i,:])
        y.append(data_y.iloc[i])
    return np.array(x), np.array(y)