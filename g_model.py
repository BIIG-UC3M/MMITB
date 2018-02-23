#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:30:24 2018

@author: pmacias
"""

import theano
from io import StringIO
import sys

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

import sys
import tempfile


def file_lines(file_path):
    with open(file_path) as f:
        content = f.readlines()
    return [x.strip() for x in content]

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
        #print n_samples,n_feats
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

def logp_gmix(mu, ch, prior, category):
    def logp_(value):
        logps =  (prior-1) + pm.MvNormal.dist(mu=mu,chol = ch).logp(value) #TODO: Performance. Esto genera las mimsas gaussians tantas veces como datos de cada categoria existan. TODO performance
        #maxi =  [ tt.sum(logps[(tt.eq(category,cat)).nonzero()] ) for cat in range(5)]
        return tt.sum(logps) 
    return logp_

def logp_gmix2(mu, ch, prior, category):
    def logp_(value):
        logps = pm.MvNormal.dist(mu=mu,chol = ch).logp(value)#TODO: Performance. Esto genera las mimsas gaussians tantas veces como datos de cada categoria existan. TODO performance
        c = tt.sum([tt.sum(logps[ (tt.eq(category,cat)).nonzero() ]) for cat in range(3) ])
        #comprobar que ahora si lo pilla
        return c
    return logp_


def logp_gmix_(mu, ch, prior, category,K):
    def logp_(value):
        logps = pm.MvNormal.dist(mu=mu,chol = ch).logp(value)#TODO: Performance. Esto genera las mimsas gaussians tantas veces como datos de cada categoria existan. TODO performance
        c = tt.sum([tt.sum(logps[ (tt.eq(category,cat)).nonzero() ]) for cat in range(K) ])
        #comprobar que ahora si lo pilla
        return c
    return logp_


# Log likelihood of normal distribution
def logp_normal(mu, tau, value):
    # log probability of individual samples
    k = tau.shape[0]
    delta = lambda mu: value - mu
    return (-1 / 2.) * (k * tt.log(2 * np.pi) + tt.log(1./det(tau)) +
                         (delta(mu).dot(tau) * delta(mu)).sum(axis=1))

# Log likelihood of Gaussian mixture distribution
def logp_gmix_priors(mus, tau, K, aux3):
    
    def logp_(value):
        aux = tt.ones((n_samples,1))
        pi = [ tt.sum(tt.eq( aux3,aux*cat), axis = 1)/8.0 for cat in range(K)]
        
        logps = [ (pi[i]-1)*2 + logp_normal(mu, tau, value) for i, mu in enumerate(mus)]
        return tt.sum(tt.stacklists(logps), axis = 0 )

        #return tt.sum(pm.math.logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

def logp_gmix_priors_chol(mus, K, aux3):
    
    def logp_(value):
        aux = tt.zeros((n_samples,1))
        pi = [ tt.sum(tt.eq( aux3,aux+cat), axis = 1)/8.0 for cat in range(K)]
        
        logps = [ ((pi[i]-1)*2 + mv.logp(value)) - tt.sum((pi[i]-1)*2 + mv.logp(value))  for i, mv in enumerate(mus)]
        return tt.sum(tt.stacklists(logps), axis = 0 )

        #return tt.sum(pm.math.logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_



def logp_gmix_priors2(mus, pi, tau, category):
    def logp_(value):
        logps = [tt.log(pi[c]) + logp_normal(mus[c], tau, value) for c in category]
          

        return tt.sum(pm.math.logsumexp(tt.stacklists(logps)[:, :n_samples], axis=0))

    return logp_

class my_Mv_normal(pm.MvNormal):
     def logp(self, value, pi):
         return super.logp(value) + (1-pi)*2




def run_normal_mv_model_prior(data, K = 3, mus = None, mc_samples = 10000, jobs = 1, n_cols = 10, n_rows = 100, neigs = 1):
    n_samples,n_feats = data.shape
    n_samples = n_cols*n_rows
    max_neigs = 4*neigs*(neigs+1)
    #print max_neigs
    to_fill = indxs_neigs(range(n_samples), n_cols=n_cols, n_rows=n_rows, n = neigs)
    inds = np.where(to_fill != -1)[0]
    to_fill = to_fill[to_fill != -1]
    aux = tt.ones(n_samples*max_neigs ) * -69 

    with pm.Model() as model:
        
        packed_L = pm.LKJCholeskyCov('packed_L', n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5))        
        L = pm.expand_packed_triangular(n_feats, packed_L)
        sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        mus = 0. if mus is None else mus

        mus = pm.Normal('mus', mu = [[10,10], [55,55], [105,105], [155,155], [205,205]], sd = 10, shape=(K,n_feats))
        #sds = pm.HalfNormal('sds',sd = 50, shape = (K,n_feats) )
        #mus = pm.Normal('mus', mu = [10,55,105,155,205], sd = sds , shape=(K,n_feats) )
        #nu = pm.Exponential('nu', 1./10, shape=(K,n_feats), testval=tt.ones((K,n_feats)) )
        #mus = pm.StudentT('mus',nu=nu, mu = [[10],[55],[105],[155],[205]], sd = 100., shape=(K,n_feats))
             
        pi = Dirichlet('pi', a=pm.floatX( [1. for _ in range(K)] ), shape=K )
        #TODO one pi per voxel
        category = pm.Categorical('category', p=pi, shape = n_samples )
        #pm.Deterministic('pri', tt.as_tensor_variable(get_prior2(category)))

        #prior = pm.Deterministic('prior',tt.stack( [tt.sum(tt.eq(category[i], category[indxs_neig(i, n_rows=73, n_cols=74)]))/8.0 for i in range(73*74) ] ))
        

        #prior = pm.Deterministic('prior',tt.sum(tt.eq(category  , category[[j for j in range(8)]].reshape( (8,1) ) )))
        
        aux2 = tt.set_subtensor(aux[inds],category[to_fill])
        prior = pm.Deterministic('prior',(tt.sum(tt.eq( aux2.reshape( (n_samples,max_neigs ) ),
                                                       category.reshape( (n_samples,1)) ), axis = 1 )+0.0)/8.0 )
        #prior2 = pm.Normal('prior2', mu = prior, sd = 0.5, shape= n_samples)
        
        
       # aux3 = tt.as_tensor_variable(pm.floatX([1,1,2,2,2,2,2,2,2,2]*100 ))
#        aux3 = tt.set_subtensor( aux3[(tt.eq(category,1)).nonzero()], 2  )
       # prior2 = pm.Deterministic('prior2', aux3 )
#        
        xs = DensityDist('x', logp_gmix(mus[category],L , prior, category), observed=data)

        
    with model:
        step2 = pm.ElemwiseCategorical(vars=[category], values=range(K) )
        #step = pm.CategoricalGibbsMetropolis(vars = [prior] )
        trace = sample(mc_samples, step = [step2], n_jobs = jobs, tune = 600)
        
        

    pm.traceplot(trace, varnames = ['mus', 'pi', 'Sigma'])
    plt.title('normal mv model 40 cols' )
    
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    #if chains > 1:
    #   print (max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values()))
    return model, mod, trace


class my_mixture(pm.Mixture):
    def _comp_modes(self):
        try:
            return tt.as_tensor_variable(self.comp_dists.mode)
        except AttributeError:
            return tt.stack([comp_dist.mode for comp_dist in self.comp_dists],axis=0)

def run_normal_mv_model_mixture(data, K = 3, mus = None, mc_samples = 10000, jobs = 1, n_cols = 10, n_rows = 100, neigs = 1):
    n_samples,n_feats = data.shape
    n_samples = n_cols*n_rows
    max_neigs = 4*neigs*(neigs+1)
    #print max_neigs
    to_fill = indxs_neigs(range(n_samples), n_cols=n_cols, n_rows=n_rows, n = neigs)
    inds = np.where(to_fill != -1)[0]
    to_fill = to_fill[to_fill != -1]
    aux = tt.ones(n_samples*max_neigs ) * -69 
    shp  = (K, n_feats)
    mus_start =  np.percentile(data,np.linspace(1,100,K), axis=0)

    with pm.Model() as model:
        
        packed_L = pm.LKJCholeskyCov('packed_L', n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5))        
        L = pm.expand_packed_triangular(n_feats, packed_L)
        sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        mus = 0. if mus is None else mus

        sds = pm.HalfNormal('sds',sd = tt.ones( shp ) * 100, shape = shp )
       
        mus = pm.Normal('mus', mu = tt.as_tensor_variable( mus_start) , sd = sds , shape=shp )

             
        pi = Dirichlet('pi', a=pm.floatX( [1. for _ in range(K)] ), shape=K )
#        #TODO one pi per voxel
        #category = pm.Categorical('category', p=pi, shape = n_samples )
        mvs = [pm.MvNormal.dist(mu = mus[i], chol = L ) for i in range(K)]
        

       
#
        #aux2 = tt.set_subtensor(aux[inds],category[to_fill])
        #prior = pm.Deterministic('prior',(tt.sum(tt.eq( aux2.reshape( (n_samples,max_neigs ) ),
        #                                               category.reshape( (n_samples,1)) ), axis = 1 )+1)/1.0 )
        
        pesos = pm.Dirichlet('pesos', a=np.ones((K,) ) )
        #obs = pm.Mixture('obs',w = pesos, comp_dists = mvs, observed = data)
        obs = my_mixture('obs',w = pesos, comp_dists = mvs, observed = data)
        
        with model:
            #step2 = pm.CategoricalGibbsMetropolis(vars=[category] )
            trace = sample(mc_samples, n_jobs = jobs, tune = 500)
            
    pm.traceplot(trace, varnames = ['mus', 'pi', 'Sigma','mvs','pesos'])
    plt.title('normal mv model 40 cols' )
    logp_simple(mus,category, aux3)
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    #if chains > 1:
    #   print (max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values()))
    return model, mod, trace

def run_normal_mv_model_mixture_DIY(data, K = 3, mus = None, mc_samples = 10000, jobs = 1, n_cols = 10, n_rows = 100, neigs = 1):
    def logp_simple(mus,category, aux3):
        def logp_(value):
            spatial_factor = 0.00
            aux = tt.ones((n_samples,)) 
            logps = tt.zeros((n_samples)) 
            sumlogps = tt.zeros((K,n_samples) ) 
            pi =  tt.sum(tt.eq( aux3,(aux*category).reshape((n_samples,1 )) ) , axis = 1)/8.0 
            #TODO son logps y sumlops siempre sustituidos en todos lo valortes
            for i,label in enumerate(range(K)):
                pi_l =  tt.sum(tt.eq( aux3,(aux*label).reshape((n_samples,1 )) ) , axis = 1)/8.0 
                sumlogps = tt.set_subtensor(sumlogps[i,:], (mus[label].logp(value)) + (pi_l - 1)*spatial_factor )
            sumlogps = tt.sum(sumlogps, axis=0)
            
            for label in range(K):
                indx = tt.eq(category,tt.as_tensor_variable(label)).nonzero()
                logps = tt.set_subtensor(logps[indx], (mus[label].logp(value)[indx]) + (pi[indx] - 1)*spatial_factor - sumlogps[indx])
            
            return logps
        return logp_
    #K = 3
    n_samples,n_feats = data.shape
    n_samples = n_cols*n_rows
    max_neigs = 4*neigs*(neigs+1)
    #print max_neigs
    to_fill = indxs_neigs(range(n_samples), n_cols=n_cols, n_rows=n_rows, n = neigs)
    inds = np.where(to_fill != -1)[0]
    to_fill = to_fill[to_fill != -1]
    aux = tt.ones(n_samples*max_neigs ) * -69 
    shp  = (K, n_feats)
    mus_start =  np.percentile(data,np.linspace(1,100,K), axis=0)
    alpha = 0.1 * np.ones((n_samples, K))
    
    with pm.Model() as model:
        
        packed_L = [pm.LKJCholeskyCov('packed_L_%d'% i, n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5) )  for i in range(K) ]     
        L = [pm.expand_packed_triangular(n_feats, packed_L[i]) for i in range(K) ]
        #sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        mus = 0. if mus is None else mus
        
        #sds = pm.Uniform('sds',lower=0., upper=150., shape = shp )
        mus = pm.Normal('mus', mu = 100. , sd = 1, shape= shp )
             
        pi = Dirichlet('pi', a = alpha, shape = (n_samples, K) )

        category = pm.Categorical('category', p=pi, shape = n_samples )
        shit_max = pm.Deterministic('shit_max',tt.max(category))
        shit_min = pm.Deterministic('shit_min',tt.min(category))

        #mvs = [MvNormal('mu_%d' % i, mu=mus[i],tau=pm.floatX(1. * np.eye(n_feats)),shape=(n_feats,)) for i in range(K)]
        mvs = [pm.MvNormal.dist(mu = mus[i], chol = L[i]) for i in range(K)]

        aux2 = tt.set_subtensor(aux[inds],category[to_fill]) 
        xs = DensityDist('x', logp_simple(mvs,category,aux2.reshape( (n_samples,max_neigs ) ) ), observed=data)
 
        with model:
            step2 = step2 = pm.ElemwiseCategorical(vars=[category], values=range(K) )
            trace = sample(mc_samples,step = step2, tune = 1000, chains = 4)
            
    pm.traceplot(trace, varnames = ['mus','sds'])
    plt.title('logp_sum_mo_alpha_700_tunes_spatial_2')
    
    mod = stats.mode(trace['category'][int(mc_samples*0.75):])
    return model, mod, trace

class Ordered(pm.distributions.transforms.ElemwiseTransform):
    name = "ordered"

    def forward(self, x):
        out = tt.zeros(x.shape)
        out = tt.inc_subtensor(out[0], x[0])
        out = tt.inc_subtensor(out[1:], tt.log(x[1:] - x[:-1]))
        return out
    
    def forward_val(self, x, point=None):
        x, = pm.distributions.distribution.draw_values([x], point=point)
        return self.forward(x)

    def backward(self, y):
        out = tt.zeros(y.shape)
        out = tt.inc_subtensor(out[0], y[0])
        out = tt.inc_subtensor(out[1:], tt.exp(y[1:]))
        return tt.cumsum(out)

    def jacobian_det(self, y):
        return tt.sum(y[1:])

def run_One_d_Model(data, K = 3, mus = None, mc_samples = 10000, jobs = 1, n_cols = 10, n_rows = 100, neigs = 1):
    def logp_simple(mus,category, aux3):
     def logp_(value):
         spatial_factor = 2
         aux = tt.ones((n_samples,)) 
         logps = tt.zeros((n_samples)) 
         sumlogps = tt.zeros((K,n_samples) ) 
         pi =  tt.sum(tt.eq( aux3,(aux*category).reshape((n_samples,1 )) ) , axis = 1)/8.0 
         #TODO son logps y sumlops siempre sustituidos en todos lo valortes
         for i,label in enumerate(range(K)):
             pi_l =  tt.sum(tt.eq( aux3,(aux*label).reshape((n_samples,1 )) ) , axis = 1)/8.0 
             sumlogps = tt.set_subtensor(sumlogps[i,:], (mus[label].logp(value)) + (pi_l - 1)*spatial_factor )
         sumlogps = tt.sum(sumlogps, axis=0)
        
         for label in range(K):
             indx = tt.eq(category,tt.as_tensor_variable(label)).nonzero()
             logps = tt.set_subtensor(logps[indx], (mus[label].logp(value)[indx]) + (pi[indx] - 1)*spatial_factor - sumlogps[indx])
         return logps
     
    n_samples,n_feats = data.shape
    n_samples = n_cols*n_rows
    max_neigs = 4*neigs*(neigs+1)
    #print max_neigs
    to_fill = indxs_neigs(range(n_samples), n_cols=n_cols, n_rows=n_rows, n = neigs)
    inds = np.where(to_fill != -1)[0]
    to_fill = to_fill[to_fill != -1]
    aux = tt.ones(n_samples*max_neigs ) * -69 
    shp  = (K, n_feats)
    mus_start =  np.percentile(data,np.linspace(1,100,K), axis=0)
    alpha = 0.1 * np.ones((n_samples, K))
    
    with pm.Model() as model:

        mu = pm.Normal('mus', 100, mus_start, shape=K, testval = mus_start, transform=Ordered())
        sd = pm.Uniform('sds',lower=0., upper=150., shape = K)
        
        #pi = Dirichlet('pi', a = alpha, shape= (n_samples, K) )
        pi = Dirichlet('pi', a = alpha, shape = K )

        category = pm.Categorical('category', p=pi, shape = n_samples )
        shit_max = pm.Deterministic('shit_max',tt.max(category))
        shit_min = pm.Deterministic('shit_min',tt.min(category))
        x = pm.NormalMixture()

class myNUTS(pm.NUTS):
    
    def astep(self, q0):
        """Perform a single NUTS iteration."""
        p0 = self.potential.random()
        start = self.integrator.compute_state(q0, p0)
        
        if np.isinf(start.energy):
            print('Go to f*** hell')
            
        if not np.isfinite(start.energy):
            raise ValueError('Bad initial energy: %s. The model '
                             'might be misspecified.' % start.energy)

        if not self.adapt_step_size:
            step_size = self.step_size
        elif self.tune:
            step_size = np.exp(self.log_step_size)
        else:
            step_size = np.exp(self.log_step_size_bar)

        if self.tune and self.m < 200:
            max_treedepth = self.early_max_treedepth
        else:
            max_treedepth = self.max_treedepth

        tree = pm.NUTS._Tree(len(p0), self.integrator, start, step_size, self.Emax)
        

        for _ in range(max_treedepth):
            direction = pm.NUTS.logbern(np.log(0.5)) * 2 - 1
            diverging_info, turning = tree.extend(direction)
            q, q_grad = tree.proposal.q, tree.proposal.q_grad

            if diverging_info or turning:
                if diverging_info:
                    self.report._add_divergence(self.tune, *diverging_info)
                break

        w = 1. / (self.m + self.t0)
        self.h_bar = ((1 - w) * self.h_bar +
                      w * (self.target_accept - tree.accept_sum * 1. / tree.n_proposals))

        if self.tune:
            self.log_step_size = self.mu - self.h_bar * np.sqrt(self.m) / self.gamma
            mk = self.m ** -self.k
            self.log_step_size_bar = mk * self.log_step_size + (1 - mk) * self.log_step_size_bar

        self.m += 1

        if self.tune:
            self.potential.adapt(q, q_grad)

        stats = {
            'step_size': step_size,
            'tune': self.tune,
            'step_size_bar': np.exp(self.log_step_size_bar),
            'diverging': bool(diverging_info),
        }

        stats.update(tree.stats())

        return q, [stats]


if __name__ == "__main__":
#    
#    old_stdout = sys.stdout
#    fp = tempfile.TemporaryFile()
#    file_name = fp.name
#    sys.stdout = open(file_name, 'w')
    
    #data,C,n_samples,K, n_feats = make_random_latent_gaussian(plot= True, cov_mat = np.array([[0.1,0.001],[0.5,1.0]] ) )
    ms = [[-1,-1],[1,1],[-3,-3]]; priors = [0.3, 0.3, 0.4 ]; cov = np.array([[1,0.25],[0.25,1]])
    data,C,n_samples,K, n_feats = make_random_latent_gaussian(plot= False, centers= ms, priors=priors, cov_mat=cov , samples= 1000 )
    data_fake  = data[:,-1:].T
    data_fake2 = data_fake+5
    data_fake = data_fake + rng.randn(len(data))*0.8
    data_fake2 = data_fake2 + rng.randn(len(data))*0.8
    #data = np.concatenate([data_fake.T, data_fake2.T, data[:,-1:]], axis = 1)
    data, n_samples, K, n_feats = image_as_data('/home/pmacias/Projects/MRI-PET_Tuberculosis/Zhang/tune_min_n.jpg')
    #model,m, trace = run_normal_mv_model_prior(data[:,:-1], mc_samples=5000, K=K)
    #model,m, trace = run_normal_mv_model_prior(data[:,:-1]class NUTS(BaseHMC):
#   mc_samples=5000, K=K, n_cols = 40, n_rows = 30, neigs=1)
    #model,m, trace = run_normal_mv_model_mixture(data[:,:-1], mc_samples=5000, K=K, n_cols = 40, n_rows = 30, neigs=1)
    #model,m, trace = run_normal_mv_model_mixture_DIY(data[:,:-1], mc_samples=1000, K=K, n_cols = 40, n_rows = 30, neigs=1)
    #model,m, trace = run_mv_model(data[:,:-1], K, n_feats=n_feats, mc_samples=50000)
    #model,m, trace = run_normal_mv_model(data[:,:-1], K=K, mc_samples=50000)
    #data,n_samples,K, n_feats = image_as_data('/home/pmacias/Projects/MRI-PET_Tuberculosis/Zhang/tune.jpg')
    #ks = np.array([4,0,1,2,3])#TODO No ojo
    #ks[trace['category'][-1]]
     

#    cmap = sns.cubehelix_palette(as_cmap=True)
#    f, ax = plt.subplots()
#    points = ax.scatter(data[:, 0], data[:, 1], c=m[1][0]/int(10000*0.75), s=(data[:,-1]+1)*20, cmap=cmap, alpha = 0.5)
#    f.colorbar(points)
    
    
    

    
    
    def logp_simple(mus,category, aux3):
        def logp_(value):
            spatial_factor = 0.0
            aux = tt.ones((n_samples,)) 
            logps = tt.zeros((n_samples)) 
            sumlogps = tt.zeros((K,n_samples) ) 
            pi =  tt.sum(tt.eq( aux3,(aux*category).reshape((n_samples,1 )) ) , axis = 1)/8.0 
            #TODO son logps y sumlops siempre sustituidos en todos lo valortes
            for i,label in enumerate(range(K)):
                pi_l =  tt.sum(tt.eq( aux3,(aux*label).reshape((n_samples,1 )) ) , axis = 1)/8.0 
                sumlogps = tt.set_subtensor(sumlogps[i,:], (mus[label].logp(value)) + (pi_l - 1)*spatial_factor )
            sumlogps = tt.sum(sumlogps, axis=0)
             
            for label in range(K):
                indx = tt.eq(category,tt.as_tensor_variable(label)).nonzero()
                logps = tt.set_subtensor(logps[indx], (mus[label].logp(value)[indx]) + (pi[indx] - 1)*spatial_factor - sumlogps[indx]) 
            
            return logps
        return logp_
    

    #K = 3
    col = data[:,2]
    data  = data[:,:-1]
    n_cols = 40
    n_rows = 30
    neigs = 1
    n_samples,n_feats = data.shape
    n_samples = n_cols*n_rows
    max_neigs = 4*neigs*(neigs+1)
    #print max_neigs
    to_fill = indxs_neigs(range(n_samples ), n_cols=n_cols, n_rows=n_rows, n = neigs)
    inds = np.where(to_fill != -1)[0]
    to_fill = to_fill[to_fill != -1]
    
    shp  = (K, n_feats)
    mus_start =  np.percentile(data,np.linspace(1,100,K), axis=0)
    alpha = 0.1 * np.ones(K)
    O = np.ones(n_samples, dtype = np.int)
    
    with pm.Model() as model:
        aux = tt.ones(n_samples*max_neigs ) * -69 
        #packed_L = [pm.LKJCholeskyCov('packed_L_%d'% i, n=n_feats, eta=2., sd_dist=pm.HalfCauchy.dist(2.5) )  for i in range(K) ]     
        #L = [pm.expand_packed_triangular(n_feats, packed_L[i]) for i in range(K) ]
        #sigma = pm.Deterministic('Sigma', L.dot(L.T))
        
        #mus = 0. if mus is None else mus
        
        sds = pm.Uniform('sds',lower=1., upper=10., shape = shp )
        mus = pm.HalfNormal('mus',  sd = sds, shape= shp )
        mus_print = tt.printing.Print('mus')(mus)
             
        pis = pm.Dirichlet('pis', a = alpha, shape = (n_samples,K))
#        pm.Potential('shit', tt.switch( tt.sum(pis) > 1.,-np.inf, 0. ) )
#        pm.Potential('shit2', tt.switch( tt.sum(pis < 0.) > 0.,-np.inf, 0. ) )

        category = pm.Categorical('category', p=pis, shape = n_samples )
        
        
#    
#        pm.Potential('shit_1', tt.switch( tt.sum( tt.eq(category, O*0 ) ) < 0.1,-np.inf, 0. ) )
#        pm.Potential('shit_2', tt.switch( tt.sum( tt.eq(category, O*1 ) ) < 0.1,-np.inf, 0. ) )
#        pm.Potential('shit_3', tt.switch( tt.sum( tt.eq(category, O*2 ) ) < 0.1,-np.inf, 0. ) )
#        pm.Potential('shit_4', tt.switch( tt.sum( tt.eq(category, O*3 ) ) < 0.1,-np.inf, 0. ) )
#        pm.Potential('shit_5', tt.switch( tt.sum( tt.eq(category, O*4 ) ) < 0.1,-np.inf, 0. ) )
#        pm.Potential('shit_6', tt.switch( tt.sum(category > 4) > 0.,-np.inf, 0. ) )
        shit_max = pm.Deterministic('shit_max',tt.max(category))
        shit_min = pm.Deterministic('shit_min',tt.min(category))
        
        pis_print = tt.printing.Print('pis')(pis)
        category_print = tt.printing.Print('category')(category)

        #mvs = [MvNormal('mu_%d' % i, mu=mus[i],tau=pm.floatX(1. * np.eye(n_feats)),shape=(n_feats,)) for i in range(K)]
        #mvs = [pm.MvNormal.dist(mu = mus[i], chol = L[i]) for i in range(K)]
        mvs = [pm.MvNormal.dist(mu = mus[i], tau=np.eye(n_feats, dtype = np.float)  ) for i in range(K)]

        aux2 = tt.set_subtensor(aux[inds],category[to_fill]) 
        xs = DensityDist('x', logp_simple(mvs,category,aux2.reshape( (n_samples,max_neigs ) ) ), observed=data)
 

        step2 = pm.ElemwiseCategorical(vars=[category], values=range(K) )
        mystep = pm.Metropolis(vars = [ mus,sds] )
        trace = sample(10000,start = pm.find_MAP(),step = [mystep,step2], tune = 3000, chains = 1, discard_tuned_samples=True, exception_verbosity='high')
            
    pm.traceplot(trace, varnames = ['mus','sds'])
    #plt.title('logp_sum_mo_alpha_700_tunes_spatial_2map')
    
    m = stats.mode(trace['category'][int(1000*0.75):])
    
    #for RV in model.basic_RVs:
    #    print(RV.name, RV.logp(model.test_point))
        
    #sys.stdout = old_stdout
    #mulines = [s for s in file_lines(file_name) if 'mus' in s]
    
    #muvals = [line.split()[-1] for line in mulines]
    #plt.plot(np.arange(0, len(muvals)), muvals)
    #plt.xlabel('proposal iteration')
    #plt.ylabel('mu value');


    
