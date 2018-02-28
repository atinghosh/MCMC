import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
# import torch
# from torch.autograd import Variable
# from numpy.linalg import inv
import theano
import theano.tensor as T
from inspect import signature
import math



class RWM(object):
    """
    This class does MH simulation, here Q(x,y)= f(y|x) ~ N(x,\epsilon),
    This can simluate from multivariate distribution

    """

    def __init__(self, fn, no_of_iteration, start, scaling_variance=1., *sigma):
        self.f = fn
        self.start = start
        if not sigma:
            self.sigma = np.identity(len(start))
        self.sigma = scaling_variance*self.sigma

        self.max = no_of_iteration
        self.dim = len(signature(fn).parameters)

    def __iter__(self):

        self.x = self.start
        #self.x = [0] * self.dim
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

        if acceptance_prob <0:
            acceptance_prob = 0

        if np.random.binomial(n=1, p=acceptance_prob) == 1:
            self.x = y
        return self.x[:self.dim]

    def sample(self, n):

        self.__new_sample = [n for n in self]
        return self.__new_sample






class MALA(object):

    def __init__(self, no_of_iteration, start, scaling_variance, *sigma):

        self.max = no_of_iteration
        self.scaling_var = scaling_variance
        self.start = np.array(start)
        #self.dim = len(start)
        if not sigma:
            self.sigma = np.identity(len(start))

    def f(self, x):
        pass

    def grad(self,x):
        pass



    def __iter__(self):
        self.x = self.start
        self.index = 0
        return self




    def __next__(self):
        self.index = self.index + 1
        if self.index > self.max:
            raise StopIteration

        mean = self.x + .5*self.scaling_var * self.grad(self.x)
        var = self.scaling_var*self.sigma
        y = multivariate_normal.rvs(mean=mean, cov=var)

        g_y_x = multivariate_normal.pdf(y, mean=mean, cov=var)
        g_x_y = multivariate_normal.pdf(self.x, mean=y+.5*self.scaling_var* self.grad(y),cov=var)
        acceptance_prob = min(1, (self.f(y)*g_x_y)/(self.f(self.x)*g_y_x))


        if np.random.binomial(n=1, p=acceptance_prob) == 1:
            self.x = y
        return self.x



if __name__ == "__main__":

    mala1 = MALA(no_of_iteration=100000, start=[1,0], scaling_variance=.01)
    # If we use too small scaling variance then we move from one point to another very slowly
    # if we use high scaling variance then we reject most of the time after reaching a high probability density point

    #Define the denisty function and gradient of the density, used theano here

    # #bivariate normal
    # x = T.vector('x')
    # #density = T.log(x[0]**3 + x[1]**3)
    # density = T.exp((-x[0] ** 2 - x[1] ** 2)/2)
    # mala1.f = theano.function([x], density)
    # #density_grad = T.grad(density, x)
    # #mala1.grad = lambda y: density_grad.eval({x:y})

    #Toy regression
    #mala1.f = lambda x : multivariate_normal.pdf([1,1],mean=x[0]**3+x[1]**3, cov=.1*np.identity(2))*multivariate_normal.pdf(x,mean=[0,0],cov=25*np.identity(2))

    x = T.vector('x')
    log_density = -((1-x[0]**3-x[1]**3)**2)/(2*.05**2) - T.dot(x,x)/50 # 50 = 2*25, so prior is N(0,25)
    mala1.f = theano.function([x], T.exp(log_density))
    log_density_grad = T.grad(log_density, x)
    mala1.grad = lambda y: log_density_grad.eval({x:y})

    # for n in mala1:
    #     print(n)


    mala_sample = [n for n in mala1]
    mala_sample_x = [n[0] for n in mala_sample]
    mala_sample_y = [n[1] for n in mala_sample]

    plt.scatter(mala_sample_x, mala_sample_y, alpha=0.1)

    plt.show()
    #print(mala_sample[-50:])









    # for n in mala1:
    #     print(n)


    # fn1 = lambda x,y: math.exp((-x**2-y**2)/2)
    # fn2 = lambda x,y: x**3+y**3
    # rwm1 = RWM(fn= fn1, no_of_iteration=1000,start = [0,0],scaling_variance=1.5)
    # rwm2 = RWM(fn= fn2, no_of_iteration=10000,start = [0,0],scaling_variance=1.5)
    #
    #
    # rwm_sample = [n for n in rwm2]
    # rwn_sample_x = [n[0] for n in rwm_sample]
    # rwn_sample_y = [n[1] for n in rwm_sample]
    # plt.plot(rwn_sample_x, rwn_sample_y)
    # plt.show()
    # for n in rwm2:
    #     print(n)



