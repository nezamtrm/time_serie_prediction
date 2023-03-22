
from keras.models import load_model
import numpy as np
from train import output

model1 = load_model('model.h5')
# predicting next 24 steps
last = np.array(output[-1])
predict = []

steps_to_predict = 24
for _ in range(steps_to_predict):
    next_prediction = model1.predict(last)
    predict.append(next_prediction)


