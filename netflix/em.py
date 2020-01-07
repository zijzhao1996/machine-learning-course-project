"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    prob = np.zeros((n, K))
    D = np.sum(np.where(X, 1, 0), axis=1)
    # construct indicator
    indicator = np.zeros(X.shape)
    for i in range (len(X)):
        for j in range (len(X[0])):
            if X[i,j] > 0 or X[i,j]<0:
                indicator[i,j]=1
            else:
                indicator[i,j]=0
                
    for i in range(n):
        # calculate the probability sample i assign to a class j
        for j in range(K):
            x = X[i, :]
            var = mixture.var[j]
            d = D[i]
            mu = mixture.mu[j,:]
            prob[i,j] = (np.power((var*2*np.pi), -d/2)* \
                        np.exp(((-1/2)* \
                        (((indicator[i]*np.transpose((x - mu))@((1/var)* \
                        np.eye(np.transpose((x - mu)).shape[0])))@(indicator[i]*(x - mu)))))))
        for k in range(K):
            P = mixture.p
            post[i, k] = prob[i,k]*P[k]/np.dot(prob[i,:],P)
    # calculate log likelinhood
    LL = np.sum(np.log(np.dot(prob,P)))
    LL
    
    return post, LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    p = np.sum(post, axis = 0)/n
    mu = np.dot(np.transpose(post),X)/(np.sum(post, axis = 0)).reshape(K,1)
    Sum = np.zeros((1,K))
    var = np.zeros((1,K))
    for i in range(n):
        for j in range(K):
            Sum[:,j] += post[i,j]*(np.linalg.norm(X[i,:]-mu[j,:]))**2
    var = np.squeeze(Sum/(d*np.sum(post, axis = 0)))
    
    return GaussianMixture(mu, var, p)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or cost - prev_cost > (1e-6)*np.abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
