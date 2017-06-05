'''
Plan:
1. Build a LSTM classifier to predict the action from historical env, rewards and actions
2. Implement the rollout to collect simulation data
3. Use rewards to update the weights using sample_weights
'''

from keras.models import Sequential
from keras.losses import binary_crossentropy
import copy
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np


'''
I think this is ready to go now,
Check list:
1. `learn()` method to learn from observation, action and reward (DONE)
2. `act()` method to suggest an action from observation (DONE)
3. agent manage the memory by itself, just need to pass `done` flag to `act()` and `learn()` (DONE)
4. implement the rollout logic to collect final results, train the again with episode and discounted reward
4.1 use agent act in the rollout




PROBLEM
How I update the model is wrong, I use model to predict rewards from (s,a) when the final reward is negative rewards, I need to update the model to not suggest the same action.
What I'm doing right now is when the result is bad, i just penalize the Q function to adjust the global reward estimation, this doesn't change the action suggestion behaviour.
SOLUTION
The problem is I'm not tracing decision make process, for parameter update, I should only update the labels that output the wrong values, for example, in this case I use argmax to retrieve the suggested action, if the outcome is negative, I should only update the predicted classes with discount factors.

'''

class LSTMAgent(object):
    def __init__(self, observation_space, action_space, timesteps):
        self.action_space = [0 for _ in range(action_space.n)]
        self.data_dim = int(observation_space.shape[0]+action_space.n)
        self.timesteps = timesteps
        self.memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]
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

        self.model.compile(loss='mean_absolute_error',
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

        Y = np.array([reward])
        this_action = self.action_space
        this_action[action] = 1
        X = self._get_SA(observation, this_action)
        seq = copy.deepcopy(self.memory)
        seq = np.concatenate((seq, [X]))
        seq = np.delete(seq, 0, axis=0)  # pop the first from memory
        X = np.array([seq])
        sample_weights = np.array([reward for _ in range(len(X))])

        self.model.fit(X, Y,
                      batch_size=batch_size, epochs=epochs,
                      validation_data=(X, Y),
                      sample_weight=sample_weights
                      )
        if done:
            self.memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]

    def _get_SA(self, observation, action):
        return np.append(observation, action)

    def _get_action_onehot(self, idx):
        ret = self.action_space[:]
        ret[idx] = 1
        return ret

    def act(self, observation, done=None):
        # argmaxQ(s,a) here, select a with largest final result
        sa_pairs = []
        for action_index in range(len(self.action_space)):
            sa_pairs.append(self._get_SA(observation, self._get_action_onehot(action_index)))
        rewards = []
        for sa in sa_pairs:
            seq = copy.deepcopy(self.memory)
            seq = np.concatenate((seq, [sa]))
            seq = np.delete(seq, 0, axis=0) # pop the first from memory
            seq = np.array([seq])
            print(seq)
            rewards.append(self.model.predict(seq))
        self.update_memory(observation, self._get_action_onehot(np.argmax(rewards)))
        action = np.argmax(rewards)
        print(action)
        if done:
            self.memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]
        return action

    def update_memory(self, observation,action):
        sa = self._get_SA(observation, action)
        self.memory = np.concatenate((self.memory, [sa]))
        self.memory = np.delete(self.memory, 0, 0)

    def roll_out(self, env, num_episode, epsilon=0.01, learn=True, greedy=False):
        '''
        Learn from roll_out, no need to return episodes now, for saving memory
        :param env:
        :param num_episode:
        :param epsilon:
        :param learn:
        :return:
        '''
        for episode in range(num_episode):
            observation = env.reset()
            env.reset()
            for t in range(1000):
                env.render()
                action = self.act(observation)
                if np.random.uniform(0,1) < episode:
                    action = self.action_space[np.random.randint(0, len(self.action_space)-1)]
                # print(observation, type(observation), action, type(action))
                observation, reward, done, info = env.step(action)
                # print(observation, type(observation), action, type(action), reward, type(reward), done, info)
                if greedy and learn:
                    self.learn(observation,action, reward)
                elif (not greedy) and learn:
                    if done:
                        self.learn(observation,action, reward)
                    else:
                        self.update_memory(observation, self._get_action_onehot(action))
                if done:
                    print(reward)
                    print("Episode finished after {} timesteps".format(t + 1))
                    break