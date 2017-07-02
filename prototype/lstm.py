from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np

data_dim = 2
timesteps = 2
num_classes = 2
train_samples = 1000
test_samples = 100
# set the sample_weights for each training sample
sample_weights = np.array([-1 for _ in range(train_samples)])

# each training sample is a 2-timestamp sequence
x_train = [[[1,1], [1,2]] for _ in range(train_samples)]
# label is the label from the input sequence
# in this setting, we don't need to output the sequence, because we are not going to re-use the sequence for another prediction
y_train = [[0,1] for _ in range(train_samples)]
# pad them
x_train = pad_sequences(x_train, maxlen=timesteps, padding='pre')
y_train = pad_sequences(y_train)

x_val = [[[1,1], [1,2]] for _ in range(test_samples)]
y_val = [[0,1] for _ in range(test_samples)]

# sample_weights = [[-1, 1]]

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=False))  # return a single vector of dimension 32
# models.add(LSTM(num_classes, return_sequences=True, activation='softmax'))  # return a single vector of dimension 32
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'],
              # sample_weight_mode='temporal'
              )



model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val),
          sample_weight=sample_weights
          )

question = np.array([[[1,1], [1,2]]])
question = pad_sequences(question, maxlen=timesteps, padding='pre')
print(question)
ret = model.predict(question, )
print(ret)