from scipy.stats import beta


def samples(a, b, success, trials, num_episodes=100):
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




'''
The conclusion is better prior requires less trails to converge.
Worse prior requires more trails to converge.
'''

successes = 3
trials = 9
# alpha, beta defines the shape of beta dist, success and trials is number of experiments.
a, b = 1, 1  # uniform
ret = samples(a, b, successes, trials, num_episodes=100)
stats(ret)

a, b = 0.5, 0.5  # convex shape prior
ret = samples(a, b, successes, trials, num_episodes=100)
stats(ret)

a, b = 1.1, 30  # 0-0.2 prior
ret = samples(a, b, successes, trials, num_episodes=100)
stats(ret)

a, b = 2, 5  # .0-0.8 prior
ret = samples(a, b, successes, trials, num_episodes=100)
stats(ret)

a, b = 2, 2  # bell shape between 0,1
ret = samples(a, b, successes, trials, num_episodes=100)
stats(ret)