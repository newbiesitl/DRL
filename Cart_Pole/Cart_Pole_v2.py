#
#
# WORKS! BUT DOESN'T SOLVE!
#
#
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym

def main():

    import gym
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    import numpy as np
    import random
    import datetime
    from gym import wrappers
    from keras.models import load_model

    def QNetwork():
        model = Sequential()
        model.add(Dense(12, input_dim=4, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mse')
        return model

    def getAction(t_observation):
        random.seed(datetime.datetime.now())
        r = random.uniform(0, 1)
        if r > Epsilon:
            return random.randint(0, 1)
        else:
            observation_array = np.array(t_observation)
            observation_array = observation_array.reshape(1, 4)
            return np.argmax(model.predict(observation_array)[0])

    # Batch train from memory -- Experience replay

    def train():
        len_memory = len(memory)
        num_actions = 2
        bsize = min(len_memory, batch_size)

        inputs = np.zeros((bsize, 4))
        targets = np.zeros((bsize, num_actions))

        mini_batch = random.sample(memory, bsize)

        for i in range(bsize):
            entry = mini_batch[i]
            state_t = entry[0]
            action_t = entry[1]  # This is action index
            reward_t = entry[2]
            state_t1 = entry[3]
            t_done = entry[4]

            s_t = np.array(state_t)
            s_t = s_t.reshape(1, 4)

            s_t1 = np.array(state_t1)
            s_t1 = s_t1.reshape(1, 4)

            inputs[i] = s_t
            targets[i] = model.predict(s_t)
            Q_s = np.max(model.predict(s_t1))

            if t_done:
                targets[i, action_t] = reward_t  # Bellman Equation
            else:
                targets[i, action_t] = reward_t + gamma * Q_s  # Bellman Equation

        model.train_on_batch(inputs, targets)

    # Storing into the memory

    def remember(param):
        memory.append(param)
        if len(param) > memory_size:
            del memory[0]

    QEpisodes = 5000
    Epsilon = 0
    memory_size = 500
    Learning_Rate = 0.4
    batch_size = 100
    gamma = 0.9
    memory = []
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'cartpole-experiment-2')
    model = QNetwork()  # Neural Network

    for i_episodes in range(QEpisodes):
        Epsilon += (1.1 / QEpisodes)  # Epsilon decay
        print
        i_episodes, Epsilon
        observation = env.reset()
        for t in range(200):
            env.render()
            temp_observation = observation[:]
            action = getAction(temp_observation)
            observation, reward, done, info = env.step(action)
            if done:
                if t < 199:
                    remember([temp_observation, action, -10, observation, done])
                else:
                    pass
                train()
                print("Episode finished after {} timesteps".format(t + 1))
                break
            else:
                remember([temp_observation, action, reward, observation, done])

    model.save('model1.h5')

    # ===================================================

    def getAction(t_observation):
        observation_array = np.array(t_observation)
        observation_array = observation_array.reshape(1, 4)
        tt = model.predict(observation_array)[0]
        # print tt
        return np.argmax(tt)

    QEpisodes = 200
    Epsilon = 1
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'cartpole-experiment-1')
    model = load_model('model5.h5')

    for i_episodes in range(QEpisodes):
        # Epsilon += (1.4 / QEpisodes)
        print(i_episodes, Epsilon)
        observation = env.reset()
        for t in range(200):
            env.render()
            temp_observation = observation[:]
            action = getAction(observation)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

#
# Driver
#
if __name__ == "__main__":
    main()
