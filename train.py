from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import numpy

numpy.random.seed(7)

# Load train capacity data-set
dataset = numpy.loadtxt("capacity-data.csv", delimiter=",")

X = dataset[:, 1] # The train trip
Y = to_categorical(dataset[:, 0:1]) # The seats taken 1 or 0 for each of 10 seats

model = Sequential()
model.add(Dense(12, input_shape=(1,)))
model.add(Dense(88))

model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mean_squared_error'])

model.fit(X, Y, epochs=10, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('my-model.h5')
