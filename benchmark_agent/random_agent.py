class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, observation_space, action_space):
        self.action_space = action_space

    def act(self, observation, done=None):
        return self.action_space.sample()

    def learn(self, env):
        # do nothing
        return