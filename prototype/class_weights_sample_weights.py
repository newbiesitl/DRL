from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

model = Sequential()
model.add(Dense(4, input_shape=(2,)))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer=optimizers.Adagrad(), loss='categorical_crossentropy')


x = np.array([[1,1], [1,1]])
y = np.array([[0,1,0,0], [0,0,0,1]])

# weight mask
weights_mask = np.array([1, 1])


# print(np.random.normal())
# exit()
# weights_mask = np.array([1])
# models.fit(x,y, epochs=1000, sample_weight=weights_mask, class_weight=class_weights, validation_data=((x,y)))
for _ in range(300):
    class_weights = {
        0: np.random.uniform(0, 1),
        1: -1,
        2: np.random.uniform(0, 1),
        3: np.random.uniform(0, 1)
    }
    model.fit(x,y, epochs=10, class_weight=class_weights, validation_data=((x,y)))

ret = model.predict(x)

print(ret)
print(sum(ret[0]))