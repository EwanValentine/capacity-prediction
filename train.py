from keras.models import Sequential
from keras.layers import Dense

import numpy

numpy.random.seed(7)

# Load train capacity data-set
dataset = numpy.loadtxt("capacity-data.csv", delimiter=",")

X = dataset[:, 1] # The train trip
Y = dataset[:, 0:1] # The seats taken 1 or 0 for each of 10 seats

model = Sequential()
model.add(Dense(24, input_dim=1))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=10, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('my-model.h5')
