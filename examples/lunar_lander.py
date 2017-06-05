import gym
from benchmark_agent.tabular_q_agent import TabularQAgent
from benchmark_agent.random_agent import RandomAgent
from prototype.lstm_agent import LSTMAgent


agent_pool = {
    'TabularQAgent': TabularQAgent,
    'RandomAgent': RandomAgent,
    'LSTMAgent': LSTMAgent
}

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.render()
# agent_1 = TabularQAgent(env.observation_space, env.action_space)
# agent_1 = agent_pool['RandomAgent'](env.observation_space, env.action_space)
print(env.action_space.n)
agent_1 = agent_pool['LSTMAgent'](env.observation_space, env.action_space, 100)
agent_1.roll_out(env, 100, greedy=False)
exit()

for i_episode in range(20):
    observation = env.reset()
    env.reset()
    for t in range(100):
        env.render()
        action = agent_1.act(observation)
        pre_observation = observation
        # print(observation, type(observation), action, type(action))
        observation, reward, done, info = env.step(action)
        # print(observation, type(observation), action, type(action), reward, type(reward), done, info)
        exit()
        agent_1.learn(env)
        if done:
            print(reward)
            print("Episode finished after {} timesteps".format(t+1))
            break