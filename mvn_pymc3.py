#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:15:04 2017

@author: pmacias
"""

import theano

import pymc3 as pm
from pymc3 import Normal, Metropolis, sample, MvNormal, Dirichlet, \
    DensityDist, find_MAP, NUTS, Slice
import theano.tensor as tt
from theano.tensor.nlinalg import det
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix(mus, pi, tau):
    def logp_(value):
        logps = [tt.log(pi[i]) + logp_normal(mu, tau, value)
                 for i, mu in enumerate(mus)]

        return tt.sum(logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

if __name__ == "__main__":
    n_samples = 100
    rng = np.random.RandomState(123)    
    ms = np.array([[-1, -1.5], [1, 1]])
    ps = np.array([0.2, 0.8])
    
    zs = np.array([rng.multinomial(1, ps) for _ in range(n_samples)]).T
    xs = [z[:, np.newaxis] * rng.multivariate_normal(m, np.eye(2), size=n_samples)
          for z, m in zip(zs, ms)]
    data = np.sum(np.dstack(xs), axis=2)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], c='g', alpha=0.5)
    plt.scatter(ms[0, 0], ms[0, 1], c='r', s=100)
    plt.scatter(ms[1, 0], ms[1, 1], c='b', s=100)
    
    from pymc3.math import logsumexp


    #Model original
    with pm.Model() as model:
        mus = [MvNormal('mu_%d' % i,
                        mu=pm.floatX(np.zeros(2)),
                        tau=pm.floatX(0.1 * np.eye(2)),
                        shape=(2,))
               for i in range(2)]
        pi = Dirichlet('pi', a=pm.floatX(0.1 * np.ones(2)), shape=(2,))
        
        xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)
        
#   
#    #Model for GMM clustering
#    with pm.Model() as model:
#        # cluster sizes
#        p = pm.Dirichlet('p', a=np.array([1., 1.]), shape=2)
#        # ensure all clusters have some points
#        p_min_potential = pm.Potential('p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))
#    
#    
#        # cluster centers
#        means = [MvNormal('mu_%d' % i,mu=pm.floatX(np.zeros(2)),tau=pm.floatX(0.1 * np.eye(2)),shape=(2,))
#               for i in range(2)]
#        # break symmetry
#        order_means_potential = pm.Potential('order_means_potential',tt.switch(means[1]-means[0] < 0, -np.inf, 0))
#    
#        # measurement error
#        sd = pm.Uniform('sd', lower=0, upper=20)
#    
#        # latent cluster of each observation
#        category = pm.Categorical('category',p=p,shape=data.shape[0])
#    
#        # likelihood for each observed value
#        points = pm.Normal('obs',
#                           mu=means[category],
#                           sd=sd,
#                           observed=data)
    
    
    
    ##For comparison with ADVI, run MCMC. 
    with model:
        start = find_MAP()
        step = Metropolis()
        trace = sample(1000, step, start=start)
        
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, c='g')
    mu_0, mu_1 = trace['mu_0'], trace['mu_1']
    plt.scatter(mu_0[-500:, 0], mu_0[-500:, 1], c="r", s=10)
    plt.scatter(mu_1[-500:, 0], mu_1[-500:, 1], c="b", s=10)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.figure()
    sns.barplot([1, 2], np.mean(trace['pi'][-5000:], axis=0),palette=['red', 'blue'])
    
    #We can use the same model with ADVI as follows.
    
#    with pm.Model() as model:
#        mus = [MvNormal('mu_%d' % i, mu=pm.floatX(np.zeros(2)), tau=pm.floatX(0.1 * np.eye(2)), shape=(2,))
#               for i in range(2)]
#        pi = Dirichlet('pi', a=pm.floatX(0.1 * np.ones(2)), shape=(2,))
#        xs = DensityDist('x', logp_gmix(mus, pi, np.eye(2)), observed=data)
#
#    with model:
#        approx = pm.fit(n=4500, obj_optimizer=pm.adagrad(learning_rate=1e-1))
#    
#        means = approx.bij.rmap(approx.mean.eval())
#        cov = approx.cov.eval()
#        sds = approx.bij.rmap(np.diag(cov)**.5)
    
