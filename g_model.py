#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:30:24 2018

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
from matplotlib.lines import Line2D
from six import iteritems
from scipy import stats

from utilities import indxs_neig,indxs_neigs

import SimpleITK

rng = np.random.RandomState(123)

def plot_data_gauss(data, centers = None):
    plt.figure()
    plt.scatter(data[:,0], data[:,1], c = data[:,-1])
    if centers is not None:
        colors = ['r','g','b','y','m', 'w','k']
        for i,c in enumerate(centers):
            plt.scatter(c[0], c[1], c = colors[i])
            

def make_random_latent_gaussian(samples = 1000, centers =[[-1, -1.5], [1, 1], [3,3]],
                                priors = [0.2, 0.6,0.2], cov_mat = None, plot = False):

    n_samples = samples
    
    ms = np.array(centers)
    ps = np.array(priors)
    
    ps_samples = (n_samples*ps).astype(np.int)
    K = len(ps_samples)

    
    C =  np.abs(rng.normal(size= (2,2))) if cov_mat is None else cov_mat

    xs = [ np.concatenate( (rng.multivariate_normal(m, C, size=z), np.ones( (z,1) ).astype(np.int)*i ), axis = 1 ) for z, m,i in zip(ps_samples, ms, range(K))]
    data = np.concatenate(xs)
    
    rng.shuffle(data)
    n_samples = len(data)
    if plot:
        plot_data_gauss(data, centers)
    return data,C,n_samples,K, data.shape[1] -1

def image_as_data(image_path):
    im = SimpleITK.ReadImage(image_path)
    im = SimpleITK.GetArrayFromImage(im).astype(np.float64)
    data = im.ravel()
    zs = np.zeros(len(data))
    zs[data > 205] = 4
    zs[ (data > 155) * (data < 205)  ] = 3
    zs[(data > 105) * (data < 155)] = 2
    zs[(data > 55) * (data < 105)] = 1
    
    data = np.array([data,data,zs]).T
    K = 5 

    return data, len(data),K, data.shape[1] - 1

def run_mv_model(data, K = 3, n_feats = 2,mus = None, mc_samples = 10000, jobs = 1):
    with pm.Model() as model:
        n_samples = len(data)
        tau = pm.Deterministic('tau', pm.floatX( tt.eye(n_feats)*10))
        mus = 0. if mus is None else mus
        mus = MvNormal('mus', mu= mus, tau= tau , shape=(K,n_feats))
        pi = Dirichlet('pi', a=pm.floatX( [1. for _ in range(K)] ), shape=K )
        category = pm.Categorical('category', p=pi, shape=n_samples)
        xs = pm.MvNormal('x', mu = mus[category], tau=tt.eye(n_feats), observed = data )
        
    with model:
        step2 = pm.ElemwiseCategorical(vars=[category], values=range(K))
        trace = sample(mc_samples, step2, n_jobs = jobs)

    pm.traceplot(trace, varnames = ['mus', 'pi', 'tau'])
    plt.title('mv model')
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    
    return model, mod, trace

def run_normal_mv_model(data, K = 3, mus = None, mc_samples = 10000, jobs = 1):
       
    with pm.Model() as model:
        n_samples,n_feats = data.shape
        print n_samples,n_feats
        packed_L = pm.LKJCholeskyCov('packed_L', n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5))        
        L = pm.expand_packed_triangular(n_feats, packed_L)
        sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        mus = 0. if mus is None else mus

        #mus = pm.Normal('mus', mu = [[10,10], [55,55], [105,105], [155,155], [205,205]], sd = 10, shape=(K,n_feats))
        mus = pm.Normal('mus', mu = mus, sd = 10., shape=(K,n_feats), testval=data.mean(axis=0))
             
        pi = Dirichlet('pi', a=pm.floatX( [1. for _ in range(K)] ), shape=K )
        #TODO one pi per voxel
        category = pm.Categorical('category', p=pi, shape=n_samples)
        xs = pm.MvNormal('x', mu = mus[category], chol=L, observed = data)
       
        
    with model:
        step2 = pm.ElemwiseCategorical(vars=[category], values=range(K))
        trace = sample(mc_samples, step2, n_jobs = jobs)

    pm.traceplot(trace, varnames = ['mus', 'pi', 'Sigma'])
    plt.title('normal mv model')
    
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    #if chains > 1:
    #   print (max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values()))
    return model, mod, trace

def logp_gmix(mu, ch, prior):
    def logp_(value):
        logps = tt.log(prior) + pm.MvNormal.dist(mu=mu,chol = ch).logp(value) #TODO: Performance. Esto genera las mimsas gaussians tantas veces como datos de cada categoria existan. TODO performance
        return logps
    return logp_

def logp_gmix2(mu, ch, prior, category):
    def logp_(value):
        logps = pm.MvNormal.dist(mu=mu,chol = ch).logp(value)#TODO: Performance. Esto genera las mimsas gaussians tantas veces como datos de cada categoria existan. TODO performance
        b = np.arange(1000, dtype = np.int)
        a = tt.sum([tt.sum(logps[jj ]) for jj in range(3) ])
        return a#logps
    return logp_


def run_normal_mv_model_prior(data, K = 3, mus = None, mc_samples = 10000, jobs = 1):
#    n_samples,n_feats = data.shape
#    to_fill = indxs_neigs(range(50*50), n_cols=50, n_rows=50)
#    inds = np.where(to_fill != -1)[0]
#    to_fill = to_fill[to_fill != -1]
    with pm.Model() as model:
        
        packed_L = pm.LKJCholeskyCov('packed_L', n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5))        
        L = pm.expand_packed_triangular(n_feats, packed_L)
        sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        mus = 0. if mus is None else mus

        #mus = pm.Normal('mus', mu = [[10,10], [55,55], [105,105], [155,155], [205,205]], sd = 10, shape=(K,n_feats))
        mus = pm.Normal('mus', mu = mus, sd = 10., shape=(K,n_feats), testval=data.mean(axis=0))
             
        pi = Dirichlet('pi', a=pm.floatX( [1. for _ in range(K)] ), shape=K )
        #TODO one pi per voxel
        category = pm.Categorical('category', p=pi, shape = n_samples)
        #pm.Deterministic('pri', tt.as_tensor_variable(get_prior2(category)))

        #prior = pm.Deterministic('prior',tt.stack( [tt.sum(tt.eq(category[i], category[indxs_neig(i, n_rows=73, n_cols=74)]))/8.0 for i in range(73*74) ] ))
        
        
        #prior = pm.Deterministic('prior',tt.sum(tt.eq(category  , category[[j for j in range(8)]].reshape( (8,1) ) )))
#        aux = tt.ones(50*50*8) * -69 #aquí un valor que no exista
#        aux2 = tt.set_subtensor(aux[inds],category[to_fill])
#        
#        prior = pm.Deterministic('prior',  pm.floatX( tt.sum(tt.eq( aux2.reshape( (50*50,8) ) , category.reshape( (50*50,1)) ), axis = 1 ))/9.0)
#        aux3 = tt.as_tensor_variable(pm.floatX( np.random.rand(2500)  ) )
#        prior2 = pm.Deterministic('prior2', aux3 )
        
        xs = DensityDist('x', logp_gmix2(mus[category],L , 1.0, category ), observed=data)

        
    with model:
        step2 = pm.ElemwiseCategorical(vars=[category], values=range(K))
        trace = sample(mc_samples, step2, n_jobs = jobs)
        
        

    pm.traceplot(trace, varnames = ['mus', 'pi', 'Sigma'])
    plt.title('normal mv model priors')
    
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    #if chains > 1:
    #   print (max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values()))
    return model, mod, trace

if __name__ == "__main__":
    #data,C,n_samples,K, n_feats = make_random_latent_gaussian(plot= True, cov_mat = np.array([[0.1,0.001],[0.5,1.0]] ) )
    ms = [[-1,-1],[1,1],[-3,-3]]; priors = [0.25, 0.25, 0.5 ]; cov = np.array([[1,0.25],[0.25,1]])
    data,C,n_samples,K, n_feats = make_random_latent_gaussian(plot= False, centers= ms, priors=priors, cov_mat=cov , samples= 1000 )
    data_fake  = data[:,-1:].T
    data_fake2 = data_fake+5
    data_fake = data_fake + rng.randn(len(data))*0.8
    data_fake2 = data_fake2 + rng.randn(len(data))*0.8
    #data = np.concatenate([data_fake.T, data_fake2.T, data[:,-1:]], axis = 1)
    #data, n_samples, K, n_feats = image_as_data('/home/pmacias/Projects/MRI-PET_Tuberculosis/Zhang/tune_n.jpg')
    model,m, trace = run_normal_mv_model_prior(data[:,:-1], mc_samples=5000, K=K)
    #model,m, trace = run_mv_model(data[:,:-1], K, n_feats=n_feats, mc_samples=50000)
    #model,m, trace = run_normal_mv_model(data[:,:-1], K=K, mc_samples=5000)
    #data,n_samples,K, n_feats = image_as_data('/home/pmacias/Projects/MRI-PET_Tuberculosis/Zhang/tune.jpg')
    #ks = np.array([4,0,1,2,3])#TODO No ojo
    #ks[trace['category'][-1]]
    
    
    
    
    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(data[:, 0], data[:, 1], c=m[1][0]/int(10000*0.75), s=(data[:,-1]+1)*20, cmap=cmap, alpha = 0.5)
    f.colorbar(points)
    
