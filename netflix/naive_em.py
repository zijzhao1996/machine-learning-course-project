"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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

    for i in range(n):
        # claculate the probability sample i assign to a class j
        for j in range(K):
            x = X[i, :]
            var = mixture.var.ravel()[j]
            d = x.shape[0]
            mu = mixture.mu[j,:]
            prob[i,j] = (np.power((var* 2*np.pi), -d/2)* \
                        np.exp(((-1/2) * \
                        (((np.transpose((x - mu))@(1/var*np.eye(d))) @ (x - mu))))))
        for k in range(K):
            P = mixture.p
            post[i, k] = prob[i,k]*P[k]/np.dot(prob[i,:],P)
    # calculate log likelinhood
    LL = np.sum(np.log(np.dot(prob,P)))
    
    return post, LL

    

    
def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

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