import os

import gym
from keras import losses
from agents.lstm_agent import LSTMAgent
from benchmark_agent.random_agent import RandomAgent

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..', )
folder_to_save = os.path.join(project_root, 'agents', 'models')
agent_pool = {
    'RandomAgent': RandomAgent,
    'LSTMAgent': LSTMAgent
}
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.render()
print('action space dimension', env.action_space.n)
mode = 'greedy'
timesteps = 30
h_dim = 32
model_name = '_'.join([env_name, mode, str(timesteps), str(h_dim)])
save_every_epoch = True
agent_1 = agent_pool['LSTMAgent'](env.observation_space, env.action_space, timesteps=30, hidden_dim=h_dim, label=model_name, loss_function='mean_absolute_percentage_error')
# None means run forever
agent_1.roll_out(env, num_episode=None, mode=mode, discount=0.99, save_every_epoch=save_every_epoch, folder_to_save=folder_to_save, train_all=True, load_saved_model=False)

# for i_episode in range(20):
#     observation = env.reset()
#     env.reset()
#     for t in range(100):
#         env.render()
#         action = agent_1.act(observation)
#         pre_observation = observation
#         # print(observation, type(observation), action, type(action))
#         observation, reward, done, info = env.step(action)
#         # print(observation, type(observation), action, type(action), reward, type(reward), done, info)
#         if done:
#             print(reward)
#             print("Episode finished after {} timesteps".format(t+1))
#             break