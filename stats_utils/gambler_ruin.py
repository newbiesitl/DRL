import numpy as np
import operator

class Gamer(object):
    def __init__(self, fortune, label='None'):
        self.fortune = fortune
        self.label = label

    def set_result(self, reward):
        self.fortune += reward
        if self.fortune <= 0:
            return False
        return True

    def roll(self, l=.0,h=1.0):
        return np.random.uniform(l, h)



def simulate(gamers, episode=None):
    '''

    :param gamers: Gamer object
    :param episode: number episode, run forever if None
    :return: result object
    '''
    if len(gamers) < 2:
        raise Exception("At least two players.")
    ret = {

    }
    while True:
        if episode is not None:
            if episode == 0:
                return ret
            else:
                episode -= 1

        buffer = [0 for _ in range(len(gamers))]
        for i, gamer in enumerate(gamers):
            buffer[i] = gamer.roll()
        winner_index, winner_value = max(enumerate(buffer), key=operator.itemgetter(1))
        results = [True for _ in range(len(gamers))]
        # update player stats
        winner_result = gamers[winner_index].set_result(len(gamers)-1)
        results[winner_index] = winner_result
        for i in range(len(gamers)):
            if winner_index != i:
                results[i] = gamers[i].set_result(-1)
        losers = [i for i, x in enumerate(results) if not x]
        if len(losers) > 0:
            return losers


if __name__ == "__main__":
    num_players = 2
    count_results = [0 for _ in range(num_players)]
    initial_fortune = [20, 20]
    for _ in range(100):
        gamer_pool = [Gamer(initial_fortune[x], str(initial_fortune[x])) for x in range(num_players)]
        ret = simulate(gamer_pool)

        for i in ret:
            count_results[i] += 1

    print('results')
    for i, v in enumerate(count_results):
        print('player {0} winning percentage: {1}'.format(str(i),str(1-v/sum(count_results))))
    print('player ')

