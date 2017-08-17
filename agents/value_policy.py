


class Prototype(object):
    def __init__(self):
        pass

    def get_value(self, state, action):
        '''
        Value function uses value network to predict final rewards. The value network can have multiple forms of inputs:
        1. State - use state to predict rewards
        2. Action - use action to predict rewards
        3. State, Action - use both State and Action to predict rewards, for example the Q function Q(s,a)
        :param state: current state
        :param action: the action with associate predictive rewards
        :return: rewards
        '''

    def get_action(self, ):