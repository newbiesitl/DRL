import gym
from benchmark_agent.tabular_q_agent import TabularQAgent
from benchmark_agent.random_agent import RandomAgent


agent_pool = {
    'TabularQAgent': TabularQAgent,
    'RandomAgent': RandomAgent
}

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.render()
# agent_1 = TabularQAgent(env.observation_space, env.action_space)
agent_1 = agent_pool['RandomAgent'](env.observation_space, env.action_space)

for i_episode in range(20):
    observation = env.reset()
    env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        agent_1.act(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        agent_1.learn(env)
        if done:
            print(reward)
            print("Episode finished after {} timesteps".format(t+1))
            break