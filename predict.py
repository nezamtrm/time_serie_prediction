
from keras.models import load_model
import numpy as np
# from train import output
import matplotlib.pyplot as plt

model1 = load_model('model_100epoch.h5')
# predicting next 24 steps
last = np.load('output.npy')
predict = []
print(last)
steps_to_predict = 24
for _ in range(steps_to_predict):
    next_prediction = model1.predict(last)
    predict.append(next_prediction)
    np.roll(predict, -1)

plt.plot(predict)


