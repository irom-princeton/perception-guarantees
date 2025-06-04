import torch
import numpy as np


def PAC_Bayes_regularizer(model, prior, N, delta, device):
    kl_div = model.calc_kl_div(prior, device)
    reg = (kl_div + np.log(2*np.sqrt(N)/delta))/2
    return reg

def kl_inv_l(q, c):
    import cvxpy as cvx
    solver = cvx.SCS
    # KL(q||p) <= c
    # try to solve: KLinv(q||c) = p

    # solve: sup  p
    #       s.t.  KL(q||p) <= c

    p_bernoulli = cvx.Variable(2)
    q_bernoulli = np.array([q, 1 - q])
    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli, p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1,
                   p_bernoulli[1] == 1.0 - p_bernoulli[0]]
    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)
    opt = prob.solve(verbose=False, solver=solver)
    return p_bernoulli.value[0]