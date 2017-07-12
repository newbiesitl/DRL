from scipy.stats import beta
from matplotlib import pyplot as plt
import numpy as np

def samples(a, b, success, trials, num_episodes=100):
    '''

    :param a: the shape param for prior dist
    :param b: the shape param for prior dist
    :param success: num success in the experiments
    :param trials:  num trails conducted
    :param num_episodes:  num samples to draw from this distribution
    :return:
    '''
    dist = beta(a+success, b+trials-success)
    episodes = num_episodes
    nums = [dist.rvs() for _ in range(episodes)]
    return nums

def stats(nums):
    avg = sum(nums)/len(nums)
    var = sum(
        [pow((x - avg), 2) for x in nums]
    )
    print(avg, var)

def plots(data, bin_size=20):
    bins = np.arange(0, bin_size, 1) # fixed bin size
    bins = bins/bin_size # normalize bins
    num_plots = len(data)
    for i, nums in enumerate(data):
        plt.subplot(num_plots, 1, i+1)
        # plot histogram
        plt.hist(nums, bins=bins, alpha=0.5)
        # hist = np.histogram(nums, bin_size)

    plt.show()



'''
The conclusion is better prior requires less trails to converge.
Worse prior requires more trails to converge.
'''

successes = 3
trials =10
# alpha, beta defines the shape of beta dist, success and trials is number of experiments.
a, b = 1, 1  # uniform
num_episodes = 2000 # num samples sampled from distribution in order to draw distribution
bin_size = 100
container = []
ret = samples(a, b, successes, trials, num_episodes=num_episodes)
container.append(ret)
stats(ret)

a, b = 0.5, 0.5  # convex shape prior
ret = samples(a, b, successes, trials, num_episodes=num_episodes)
container.append(ret)
stats(ret)

a, b = 1.1, 30  # 0-0.2 prior
ret = samples(a, b, successes, trials, num_episodes=num_episodes)
container.append(ret)
stats(ret)

a, b = 2, 5  # .0-0.8 prior
ret = samples(a, b, successes, trials, num_episodes=num_episodes)
container.append(ret)
stats(ret)

a, b = 2, 2  # bell shape between 0,1
ret = samples(a, b, successes, trials, num_episodes=num_episodes)
container.append(ret)
stats(ret)

plots(container, bin_size=bin_size)