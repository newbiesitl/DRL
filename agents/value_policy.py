


class Prototype(object):
    def __init__(self):
        pass

    def value_sampler(self, state, action):
        '''
        Value network to predict final rewards. The value network can have multiple forms of inputs:
        1. State - use state to predict rewards
        2. Action - use action to predict rewards
        3. State, Action - use both State and Action to predict rewards, for example the Q function Q(s,a)
        :param state: current state
        :param action: the action with associate predictive rewards
        :return: rewards (1-D array)
        '''

    def policy_sampler(self, state):
        '''
        Policy network to sample actions based on state, use as approximation to reduce search complexity
        :param state:
        :return:
        '''