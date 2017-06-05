'''
Plan:
1. Build a LSTM classifier to predict the action from historical env, rewards and actions
2. Implement the rollout to collect simulation data
3. Use rewards to update the weights using sample_weights
'''

from keras.models import Sequential
import copy
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class LSTMAgent(object):
    def __init__(self, data_dim, timesteps, action_space):
        self.action_space =action_space
        self.data_dim = data_dim
        self.timesteps = timesteps
        self.memory = pad_sequences([], maxlen=self.timesteps, padding='pre')

        # this Q function predict accumulative reward from state and action Q(s,a)
        # regression model output unbounded / normalized score
        # expected input data shape: (batch_size, timesteps, data_dim)
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True,
                       input_shape=(self.timesteps, self.data_dim)))  # returns a sequence of vectors of dimension 32
        # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(32, return_sequences=False))  # return a single vector of dimension 32
        # model.add(LSTM(num_classes, return_sequences=True, activation='softmax'))  # return a single vector of dimension 32
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'],
                          # sample_weight_mode='temporal'
                          )

    def learn(self, observation, action, reward, done=False, batch_size=1, epochs=1):
        '''
        train model with (s,a) to Reward mapping
        actions are stored in self.action_space
        create X by concatenate env with action
        use final result as Y
        :param observation:
        [ 0.00240507  0.92378398  0.12652748 -0.5739521  -0.00432121 -0.0594537   0. 0. ]
         <class 'numpy.ndarray'>  -2.17269424478 <class 'numpy.float64'> False {}
        :param batch_size:
        :param epochs:
        :return:
        '''

        Y = reward
        X = self.get_SA(observation, action)
        seq = copy.deepcopy(self.memory)
        seq.append(X)
        seq.pop(0)
        sample_weights = reward
        X = seq
        self.model.fit([X], [Y],
                      batch_size=batch_size, epochs=epochs,
                      validation_data=([X], [Y]),
                      sample_weight=sample_weights
                      )
        if done:
            self.memory = pad_sequences([], maxlen=self.timesteps, padding='pre')

    def get_SA(self, observation, action):
        return np.append(observation, action)

    def act(self, observation, done=None):
        # argmaxQ(s,a) here, select a with largest final result
        sa_pairs = [self.get_SA(observation, action) for action in self.action_space]
        seqs=[]
        for sa in sa_pairs:
            seq = copy.deepcopy(self.memory)
            seq.append(sa)
            seq.pop(0)
            seqs.append(seq)
        rewards = self.model.predict(seqs)
        action = self.action_space[np.argmax(rewards)]
        if done:
            self.memory = pad_sequences([], maxlen=self.timesteps, padding='pre')
        return action

