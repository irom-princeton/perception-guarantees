import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.linalg import expm
import scipy.optimize as opt
from scipy.sparse import bmat

import matplotlib.pyplot as plt

from shapely.geometry import box, Polygon, LineString
from shapely.ops import unary_union


import pickle

[k1, k2, A, B, R, BRB] = pickle.load(open('planning/sp_var.pkl','rb'))

# WORLD FUNCTIONS
def turn_box(b_list):
    '''converts format of bounding boxes'''
    box_list = []
    for b in b_list:
        box_list.append(box(b[0][0],b[0][1],b[1][0],b[1][1]))
    return box_list

def non_det_filter(xbar_prev, xhat_now):
    '''
    implements non-deterministic filter
    inputs: xbar_prev, xhat_now
    xbar_prev: previous estimate of occ_space
    xhat_now: current estimate of occ_space
    
    output: xbar_now, updated estimate of occ_space
    '''
    xhat_all = unary_union(turn_box(xhat_now))
    return xhat_all.intersection(xbar_prev)
    

# DYNAMICS FUNCTIONS: Reachability
def gramian(t):
    '''computes weighted controllability gramian'''
    def gdot(t,g0):
        g = g0.reshape(4,4)
        res = A@g + g@A.T + BRB
        return res.reshape(-1)
    return solve_ivp(gdot,[0,t],np.zeros(16)).y[:,-1].reshape(4,4)

def cost_optimal(state0, state1, r):
    def cost(t):
        x = state1-expm(A*t)@state0
        G = gramian(t)
        c =t+x.T@np.linalg.inv(G)@x
        return c
    def cost_dot(t):
        x = state1-expm(A*t)@state0
        G = gramian(t)
        d = np.linalg.inv(G)@x
        return 1-2*(A@state1).T@d-d.T@BRB@d
    
    t_star = opt.root(cost_dot, 0.1).x[0]
    return cost(t_star), t_star

def forward_box(state, r, vx_range, vy_range):
    '''rough filtering of forward-reachable set'''
    vxmax = max(vx_range)
    vymax = max(vy_range)
    vxmin = min(vx_range)
    vymin = min(vy_range)
    # cost >= time, so t_max = r
    xmax,ymax = state[0:2] + r*np.array([vxmax,vymax])
    xmin,ymin = state[0:2] + r*np.array([vxmin,vymin])
    return xmax,ymax,xmin,ymin

def backward_box(state, r, vx_range, vy_range):
    '''rough filtering of back-reachable set'''
    vxmin = min(vx_range)
    vymin = min(vy_range)
    vxmax = max(vx_range)
    vymax = max(vy_range)
    xmax,ymax = state[0:2] - r*np.array([vxmin,vymin])
    xmin,ymin = state[0:2] - r*np.array([vxmax,vymax])
    return xmax,ymax,xmin,ymin

def filter_reachable(state: np.ndarray, state_set: list, r, vx_range, vy_range, direction: str, dt: float):
    """
    Filter reachable states

    Args:
        state (np.ndarray): Current state
        state_set (list): Set of state indices to filter
        r: cost threshold
        vx_range (list): Range of vx
        vy_range (list): Range of vy
        direction (str): 'F' for forward, 'B' for backward
        dt: time resolution to compute trajectory

    Returns:
        state_set_filtered (list): Filtered set of state indices
        cost_set_filtered (list): costs corresponding to filtered state indices
        time_set_filtered (list): times corresponding to filtered state indices
    """
    if direction == 'F':
        box = forward_box
    elif direction == 'B':
        box = backward_box

    xmax, ymax, xmin, ymin= box(state,r,vx_range,vy_range)
    state_set_filtered = []
    cost_set_filtered = []
    time_set_filtered = []
    traj_set_filtered = []
    for idx in range(len(state_set)):
        state_i = state_set[idx]
        if np.any(state_i != state):
            if xmin <= state_i[0] <= xmax and ymin <= state_i[1] <= ymax:
                if direction == 'F':
                    cost, time = cost_optimal(state, state_i, r)
                elif direction == 'B':
                    cost, time = cost_optimal(state_i, state, r)
                if cost <= r:
                    if direction == 'F':
                        x,u = gen_trajectory(state, state_i, time, dt)
                    elif direction == 'B':
                        x,u = gen_trajectory(state_i, state, time, dt)
                    # if (np.all(min(vx_range)-0.6<=u[:,0]) and np.all(u[:,0]<=max(vx_range)+0.6) 
                    #     and np.all(min(vy_range)-0.6<=u[:,1]) and np.all(u[:,1]<=max(vy_range)+0.6)
                    #     and np.all(min(vx_range)-0.4<=x[:,2]) and np.all(x[:,2]<=max(vx_range)+0.4)
                    #     and np.all(min(vy_range)-0.4<=x[:,3]) and np.all(x[:,3]<=max(vy_range)+0.4)
                    #     and np.all(0<=x[:,0]) and np.all(x[:,0]<=8) and np.all(0<=x[:,1]) and np.all(x[:,1]<18)):
                    state_set_filtered.append(idx)
                    cost_set_filtered.append(cost)
                    time_set_filtered.append(time)
                    traj_set_filtered.append((x,u))

    return state_set_filtered, cost_set_filtered, time_set_filtered, traj_set_filtered


# DYNAMICS FUNCTIONS: Trajectory
def gen_trajectory(s0, s1, tau, dt):
    '''
    Generates the optimal trajectory connecting two points in state space
    
    Inputs:
        s0: initial state
        s1: final state
        tau: optimal connection time
        dt: time step
    '''
    sbar = expm(A*tau)@s0
    Ginv = np.linalg.inv(gramian(tau))
    d = Ginv@(s1 - sbar)
    
    block_mat = bmat([[A,BRB.T],[None,-A.T]]).toarray()
    block_vec = np.hstack([s1,d])
    
    def xydot(t,xy):
        return block_mat@xy

    waypoints = solve_ivp(xydot,[tau,0],block_vec,t_eval = np.arange(tau, 0, -dt)).y
    x_waypoints = waypoints[0:4,:].T
    u_waypoints = (np.linalg.inv(R)@B.T@waypoints[4:,:]).T

    return x_waypoints[::-1], u_waypoints[::-1]
