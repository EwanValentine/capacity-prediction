from keras.models import load_model
import numpy

model = load_model('my-model.h5')

X = numpy.array([1])

res = model.predict(X)
print(res)
