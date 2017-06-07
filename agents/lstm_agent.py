'''
Plan:
1. Build a LSTM classifier to predict the action from historical env, rewards and actions
2. Implement the rollout to collect simulation data
3. Use rewards to update the weights using sample_weights
'''

from keras.models import Sequential, model_from_json
from keras.losses import binary_crossentropy
from keras import metrics
import copy, os
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np



'''
This is Q-Learning, we don't need off policy update, there's no approximation here, we are doing exhaustive search over action space

TODO @charles
1. Save/Load
2. adaptive
'''

class LSTMAgent(object):
    def __init__(self, observation_space, action_space, timesteps, label, hidden_dim=32, discount_factor=1):
        self.action_space = [0 for _ in range(action_space.n)]
        self.data_dim = int(observation_space.shape[0]+action_space.n)
        self.timesteps = timesteps
        self.label = label
        self.memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]
        # this Q function predict accumulative reward from state and action Q(s,a)
        # regression model output unbounded / normalized score
        # expected input data shape: (batch_size, timesteps, data_dim)
        self.model = Sequential()
        self.model.add(LSTM(hidden_dim, return_sequences=True,
                       input_shape=(self.timesteps, self.data_dim)))  # returns a sequence of vectors of dimension 32
        # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(hidden_dim, return_sequences=False))  # return a single vector of dimension 32
        # model.add(LSTM(num_classes, return_sequences=True, activation='softmax'))  # return a single vector of dimension 32
        self.model.add(Dense(1, activation='linear'))
        self.discount = discount_factor
        self.model.compile(loss='mean_absolute_error',
                          optimizer='adam',
                          metrics=['mean_absolute_error'],
                          # sample_weight_mode='temporal'
                          )

    def learn(self, seq, rewards, batch_size=1, epochs=2, ratio=0.8):
        # for non-greedy, use the final rewards for the entire sequence
        # for greedy, use current reword for each time step in seq
        memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]
        # sequence can be longer than time steps
        X = []
        Y = []
        print('final rewards', sum(rewards)/len(rewards))
        print('rewards history', rewards)
        for i in range(len(seq)):
            observation, action_idx = seq[i]
            action = self.action_space[:]
            # one-hot action vector
            action[action_idx] = 1
            _X = self._get_SA(observation, action)
            reward = rewards[i] if rewards[i] > 0 else rewards[i] * 1
            Y.append(np.array(reward))

            memory = np.concatenate((memory, [_X])) # append new event in memory
            memory = np.delete(memory, 0, axis=0)  # pop the first from memory
            _X = np.array(memory)
            # print(_X.shape)
            X.append(_X)
        # for i in range(len(X)):
        #     print(X[i], Y[i])
        divider = int(len(X)*ratio)
        if divider == 0:
            x_train = np.array(X)
            y_train = np.array(Y)
            x_test = np.array(X)
            y_test = np.array(Y)
        else:
            x_train = np.array(X[:divider])
            y_train = np.array(Y[:divider])
            x_test = np.array(X[divider:])
            y_test = np.array(Y[divider:])
        # print(x_train.shape)
        # apply discount here
        sample_weights = []

        tmp = []
        for i in range(len(sample_weights)):
            tmp.append(sample_weights[i] * y_train[i])
        print('discount rewards', tmp)
        acc = 1
        for _ in range(len(x_train)):
            sample_weights.insert(0, acc)
            acc *= self.discount
        sample_weights = np.array(sample_weights)
        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       validation_data=(x_test, y_test),
                       sample_weight=sample_weights
                       )


    def online_learning(self, observation, action, reward, done=False, batch_size=1, epochs=1):
        raise NotImplemented()

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
            reward = self.model.predict(seq)
            # print(sa, reward)
            rewards.append(reward)
        self.update_memory(observation, self._get_action_onehot(np.argmax(rewards)))
        action = np.argmax(rewards)
        if done:
            self.memory = pad_sequences([[[0 for _ in range(self.data_dim)]]], maxlen=self.timesteps, padding='pre')[0]
        return action

    def update_memory(self, observation,action):
        sa = self._get_SA(observation, action)
        self.memory = np.concatenate((self.memory, [sa]))
        self.memory = np.delete(self.memory, 0, 0)

    def roll_out(self, env, num_episode, epsilon=0.01, discount=1, mode='greedy', save_every_epoch=False, folder_to_save='.'):
        '''

        :param env:  env object
        :param num_episode:  if None run forever
        :param epsilon:  epsilon greedy for random exploration
        :param discount: the discount factor
        :param mode:
            `greedy`: use heuristic and final reward
            `global`: only use final reward
            `heuristic`: use heuristic for non termination states, use final reward for termination state
        :save_every_epoch: whether save current mode after each epoch
        :return:
        '''
        self.discount = discount
        episode = 0
        reset = True if num_episode is None else False
        num_episode = num_episode if num_episode is not None else 100
        while episode < num_episode:
            if reset:
                episode = 0
            episode += 1
            observation = env.reset()
            env.reset()
            episodes = []
            rewards = []
            action_history = []
            for t in range(1000):
                env.render()
                action = self.act(observation)
                pre_observation = observation
                if np.random.uniform(0,1) < epsilon:
                    ind = np.random.randint(0, len(self.action_space))
                    print('exploration action:', ind)
                    action = ind
                # print(observation, type(observation), action, type(action))
                observation, reward, done, info = env.step(action)
                action_history.append(action)

                # what i'm doing here is to store the episode until the end
                # when episode finish update the model with entire episode with eligibility trace and discount
                episodes.append([pre_observation, action])
                rewards.append(reward)
                self.update_memory(pre_observation, self._get_action_onehot(action))

                if done:
                    # final training after end of episode
                    if mode == 'global':
                        # use final rewards as label
                        rewards = [rewards[-1] for _ in range(len(rewards))]
                    if mode == 'heuristic':
                        rewards = rewards # use heuristic and final result
                    if mode == 'greedy':
                        rewards = [(x+rewards[-1])/2 for x in rewards]
                    else:
                        raise Exception('unknown mode, supported mode are {0}'.format(' '.join(['global', 'greedy', 'heuristic'])))
                    print('actions', action_history)
                    self.learn(episodes, rewards, batch_size=10, epochs=5)
                    if save_every_epoch:
                        self.save(folder_to_save, self.label)
                    print(reward)
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def load(self, folder_path, model_name):
        arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'arch']), 'json']))
        weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'weights']), 'json']))
        # Load autoencoder architecture + weights + shapes
        json_file = open(arch_file, 'r')  # read architecture json
        autoencoder_json = json_file.read()
        json_file.close()
        self.model = model_from_json(autoencoder_json)  # convert json -> model architecture
        self.model.load_weights(weights_file)  # load model weights
        self.model_input_shape = self.model.input_shape  # set input shape from loaded model
        self.model_output_shape = self.model.output_shape  # set output shape from loaded model




    def save(self, folder_path, model_name):
        arch_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder', 'arch']), 'json']))
        weights_file = os.path.join(folder_path, '.'.join(['_'.join([model_name, 'encoder', 'weights']), 'json']))
        # Save autoencoder model arch + weights
        with open(arch_file, "w+") as json_file:
            json_file.write(self.model.to_json())  # arch: json format
        self.model.save_weights(weights_file)  # weights: hdf5 format

