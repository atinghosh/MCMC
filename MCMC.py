import numpy as np
import math
from inspect import signature
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from numpy.linalg import inv




class MetropollisHastings2(object):
    """
    This class does MH simulation, here Q(x,y)= f(y|x) ~ N(x,\epsilon),
    This can simluate from multivariate distribution

    """

    def __init__(self, fn, sigma, no_of_iteration):
        self.f = fn
        self.sigma = sigma
        self.max = no_of_iteration
        self.dim = len(signature(fn).parameters)

    def __iter__(self):
        self.x = [0] * self.dim
        self.index = 0
        return self

    def __next__(self):

        self.index = self.index + 1
        if self.index > self.max:
            raise StopIteration

        y = multivariate_normal.rvs(self.x, self.sigma)
        if self.dim == 1:
            y = [y]
        acceptance_prob = min(1., self.f(*y) / self.f(*self.x))
        if np.random.binomial(n=1, p=acceptance_prob) == 1:
            self.x = y
        return self.x[:self.dim]

    def sample(self, n):

        self.__new_sample = [n for n in self]
        return self.__new_sample


class MetropollisHastings(object):
    """
    This class does MH simulation, here Q(x,y)= f(y|x) ~ N(x,\epsilon)

    """

    def __init__(self, univatiate_fn, sigma, no_of_iteration, start):
        self.f = univatiate_fn
        self.sigma = sigma
        self.max = no_of_iteration
        self.start = start

    def __iter__(self):
        self.x = self.start
        self.index = 0
        return self

    def __next__(self):
        self.index = self.index + 1
        if self.index > self.max:
            raise StopIteration

        y = self.sigma * np.random.randn() + self.x
        acceptance_prob = min(1, self.f(y) / self.f(self.x))
        if np.random.binomial(n=1, p=acceptance_prob) == 1:
            self.x = y
        return self.x

    def sample(self, n, generate_new=True):
        if hasattr(self, 'new_sample') and len(self.new_sample) >= n and not generate_new:
            return self.new_sample

        self.new_sample = [n for n in self]
        return self.new_sample

    def expectation(self, phi, total_sample=10000, burn_in=1000, generate_new=True):

        if len(self.new_sample) >= total_sample and not generate_new:
            final_sample = self.new_sample[burn_in:]
            final_sample = [phi(n) for n in final_sample]

        else:
            self1 = self
            self1.max = total_sample
            new_sample = [n for n in self1]
            final_sample = new_sample[burn_in:]
            final_sample = [phi(n) for n in final_sample]

        return np.mean(final_sample)







if __name__ == "__main__":

    f1 = lambda x: math.exp(-(x ** 2 / (2 + math.sin(x))))
    mh_1 = MetropollisHastings(f1, 10, 100000, start=0)

    a_sample = mh_1.sample(1000, generate_new = True)
    plt.plot(a_sample[:200])
    #print(a_sample)
    print(mh_1.expectation(phi = lambda x:x, generate_new = False,burn_in= 20000))

    #plt.hist(a_sample,1000)
    plt.show()

    # print(mh_1.sample(10000, generate_new=False))
    # print(mh_1.expectation(phi=lambda x: x, generate_new = False))






# def expectation(phi, distribution, total_sample=10000, burn_in=1000, sigma=.1, start=0):
#     sample = [n for n in MetropollisHastings(distribution, sigma, total_sample, start)]
#     final_sample = sample[burn_in:]
#     final_sample = [phi(n) for n in final_sample]
#     return np.mean(final_sample)
# for n in MetropollisHastings(lambda x: math.exp(-(x**2/(2+math.sin(x)))),.001,100,start=-5):
#     print(n)

# for n in MetropollisHastings(lambda x: math.exp(-x**2/2),.01,100000000,start=4):
#     print(n)



