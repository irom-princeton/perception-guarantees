import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.linalg import expm
import scipy.optimize as opt
from scipy.sparse import bmat

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from shapely.geometry import box, Polygon, LineString
from shapely.ops import unary_union


import pickle
import os

[k1, k2, A, B, R, BRB] = pickle.load(open('planning/sp_var.pkl','rb'))

# WORLD FUNCTIONS
def overlap(a, b): # 2d implementation
    '''
    returns rectangle overlap between two rectangles
    inputs: a, b, rectangles [[xmin, ymin], [xmax, ymax]] (2d)
    returns: overlapped rectangle, or a, b if no overlap
    '''

    ab1x = max(a[0,0], b[0,0])
    ab1y = max(a[0,1], b[0,1])

    dx = min(a[1,0], b[1,0]) - ab1x
    dy = min(a[1,1], b[1,1]) - ab1y
    if (dx>=0) and (dy>=0):
        return np.array([[[ab1x, ab1y], [ab1x+dx, ab1y+dy]]])
    else:
        # return np.array([a, b])
        return None

def turn_box(b_list):
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
    '''old method
    ab_all = []
    for a in xbar_prev:
        for b in xhat_now:
            # print(ab)
            ab = overlap(a, b)
            if np.all(a==ab) and np.all(b==ab):
                # a,b same box, keep one
                ab_all.append(ab)
            elif (ab is None):
                # no overlap
                continue
            else:
                ab_all.append(ab)
    if len(ab_all) == 0:
        return []
    else:
        ab_all_unique = []
        for ab in ab_all:
            if not any(np.all(ab == x) for x in ab_all_unique):
                ab_all_unique.append(ab)
        xbar_now = np.concatenate(ab_all)
        return xbar_now'''
    # using shapely
    xhat_all = unary_union(turn_box(xhat_now))
    return xhat_all#.intersection(xbar_prev)
    
        

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
    
    # t_star = opt.minimize(cost, r/8, bounds = [(0,r)], method = 'trust-constr', options={'gtol': 1e-3, 'maxiter': 10}).x[0]
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
    
    rip = 0
    yay = 0

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
                    # if (np.all(np.logical_or.reduce((np.logical_and(u[:,0] >= -2, u[:,0] <= -0.19),
                    #                                 u[:,0]==0,
                    #                                 np.logical_and(u[:,0] >= 0.19, u[:,0] <= 2)))) 
                    #     and np.all(np.logical_or(u[:,0]==0,
                    #                              np.logical_and(u[:,1] >= 0.19, u[:,1] <= 2))) 
                    #     and np.all(min(vx_range)-0.3<=x[:,2]) and np.all(x[:,2]<=max(vx_range)+0.3)
                    #     and np.all(min(vy_range)-0.3<=x[:,3]) and np.all(x[:,3]<=max(vy_range)+0.3)
                    #     and np.all(0<=x[:,0]) and np.all(x[:,0]<=8) and np.all(0<=x[:,1]) and np.all(x[:,1]<18)):
                    #     state_set_filtered.append(idx)
                    #     cost_set_filtered.append(cost)
                    #     time_set_filtered.append(time)
                    #     traj_set_filtered.append((x,u))
                    #     yay += 1
                    # else:
                    #     rip += 1
                    state_set_filtered.append(idx)
                    cost_set_filtered.append(cost)
                    time_set_filtered.append(time)
                    traj_set_filtered.append((x,u))
                    yay += 1
    print('rip: ',rip, 'yay: ', yay)
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

def show_trajectory(ax, s0, s1, tau, dt, c_='gray', linewidth_=0.5):
    x_waypoints,_ = gen_trajectory(s0, s1, tau, dt)
    M = np.zeros((4, int(np.abs(np.ceil(tau/dt)))))
    for i in range(len(x_waypoints)):
        M[:, i] = x_waypoints[i]
    ax.plot(M[0, :], M[1, :], c=c_, linewidth=linewidth_)

def gen_path(s0, s1, dt):
    '''Generates straight path without dynamics'''
    dx, dy = s1[0] - s0[0], s1[1] - s0[1]
    yaw = np.arctan2(dy, dx)
    d = np.hypot(dx, dy)
    # print(d,dx)
    steps = np.arange(0, d, dt).reshape(-1, 1)
    pts = s0[0:2] + steps * np.array([np.cos(yaw), np.sin(yaw)])
    return np.vstack((pts, s1[0:2]))


# SUBGOAL FUNCTIONS
def ccw(A,B,C):
    # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    # Return true if line segments AB and CD intersect

    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line_intersection(line1, line2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def find_frontier(xbar_now, world_box, start, FoV, polygon=False):
    '''
    Finds the unknown frontier
    
    Inputs:
        xbar_now: current estimate of occ_space
        world_box: world boundary
        start: current state
        FoV: field of view
    
    Output:
        frontier: dictionary of rays and corresponding segments
    '''
    ray_left = np.pi/2+FoV/2
    ray_right = np.pi/2-FoV/2
    box_vertices = {}
    vertex_rays = {'left':ray_left,'right':ray_right}
    rays = []
    frontier = {}
    ray_endbox = {}
    ray_startbox = {}
    scaling = np.linalg.norm(world_box[0]-world_box[1])
    
    world_vertices = [world_box[0],np.array([world_box[1][0],world_box[0][1]]),world_box[1],np.array([world_box[0][0],world_box[1][1]])]
    world_edges = np.array([[world_vertices[i],world_vertices[(i+1)%4]] for i in range(4)])
    
    # find all rays corresponding to all vertices
    for i in range(len(xbar_now.geoms)):
        obstacle = xbar_now.geoms[i]
        vertx,verty = obstacle.exterior.xy
        box_vertices[i] = np.array([vertx,verty]).T
        for j in range(len(vertx)):
            vertex = np.array([vertx[j],verty[j]])
            ray = np.mod(np.arctan2(vertex[1]-start[1],vertex[0]-start[0]),2*np.pi)
            if ray_right <= ray <= ray_left: # vertex is in FoV
                vertex_rays[vertex.tobytes()] = ray
                ray_endbox[ray] = i
    # discard rays in the way
    for vert, ray in vertex_rays.items():
        show = True
        if type(vert) == bytes:
            vertex = np.frombuffer(vert)
        else:
            vertex = np.array(start[0:2])
        ray_line = np.array([start[0:2],start[0:2]+scaling*np.array([np.cos(ray),np.sin(ray)])])
        
        segments = []
        # find intersection with world boundary
        for world_edge in world_edges:
            if intersect(world_edge[0],world_edge[1],ray_line[0],ray_line[1]):
                intersection = line_intersection(world_edge, ray_line)
                segment = np.array([vertex, intersection])
        if np.linalg.norm(segment[0]-segment[1]) < 1e-5:
            segment = np.array([vertex-np.array(1e-5*np.cos(ray),1e-5*np.sin(ray)), 
                                intersection+1e-5*np.array([np.cos(ray),np.sin(ray)])])
        segments.append(segment)

        # for this ray, look at each box
        for box, vertices in box_vertices.items():
            if xbar_now.geoms[box].geom_type != 'Polygon':
                continue
            b = xbar_now.geoms[box].exterior.coords
            edges = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
            if np.any(abs(vertices - vertex)<1e-5): # same box
                # make sure ray doesn't intersect with its own box
                for edge in edges:
                    edge = np.array(edge.coords)
                    if np.any(np.all(edge == vertex, axis = 1)): # this edge is for this vertex
                        continue
                    else:
                        if intersect(edge[0],edge[1],ray_line[0],ray_line[1]):
                            intersection = line_intersection(edge, ray_line)
                            show = False
            else: # different box
                # discard rays blocked by other boxes
                # get edges
                for edge in edges:
                    edge = np.array(edge.coords)
                    # find intersection
                    if intersect(edge[0],edge[1],ray_line[0],ray_line[1]):
                        intersection = line_intersection(edge, ray_line)
                        # compute distance
                        dist_to_edge = np.linalg.norm(np.array(intersection)-np.array(start[0:2]))
                        dist_to_vertex = np.linalg.norm(vertex-np.array(start[0:2]))
                        if dist_to_edge <= dist_to_vertex:
                            show = False
                        else:
                            segments.append(np.array([vertex, intersection]))
                        
        if ray not in rays and show == True:
            rays.append(ray)
            segments_len = np.array([np.linalg.norm(segment[0]-segment[1]) for segment in segments])
            # if np.min(segments_len) > 1e-5:
            frontier[ray] = segments[np.argmin(segments_len)]
            
    if polygon:
        return frontier, ray_startbox, ray_endbox
    else:
        return frontier

def find_candidates(frontier, radius):
    '''find subgoal candidates'''
    candidates = []
    for ray_angle, segment in frontier.items():
        # segment: [[x1,y1],[x2,y2]]
        segment_len = np.linalg.norm(segment[0]-segment[1])
        num_segments = int(np.floor(0.5*segment_len/radius))
        for i in range(num_segments):
            candidates.append(segment[0] + np.array([2*(i+1/2)*radius*np.cos(ray_angle),2*(i+1/2)*radius*np.sin(ray_angle)]))
    return candidates

def plot_frontier(occ_space, world_box, start, FoV,radius):
    segments_dict = find_frontier(occ_space, world_box, start, FoV)
    candidates = find_candidates(segments_dict, radius)
    fig, ax = plt.subplots()
    ax.set_xlim([-0.1,world_box[1][0]+0.1])
    ax.set_ylim([-0.1,world_box[1][1]+0.1])
    for geom in occ_space.geoms:    
        xs, ys = geom.exterior.xy    
        ax.fill(xs,ys, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5))
    for segment in segments_dict.values():
        ax.plot([segment[0,0],segment[1,0]],[segment[0,1],segment[1,1]])
    if len(candidates) > 0:
        ax.scatter(*zip(*candidates))
    plt.show()

# OCCLUSION FUNCTIONS
    
def find_box_idx(ray_x_inter,ray_y_inter,ray_,box_x,box_y):
    box_idx = None
    if len(ray_x_inter) != 0 and len(ray_y_inter) == 0: # vertical middle
    # consecutive indices means ray is inbetween the two vertices
        start_idx = np.where(np.diff(ray_x_inter) == 1)[0]+1
        for i in start_idx:
            # check if ray1_y is inbetween the two vertices
            if (box_y[ray_x_inter[i]] <= ray_[1] <= box_y[ray_x_inter[i]-1] or 
                box_y[ray_x_inter[i]-1] <= ray_[1] <= box_y[ray_x_inter[i]]):
                box_idx = ray_x_inter[i]
                break
    elif len(ray_y_inter) != 0 and len(ray_x_inter) == 0: # horizontal middle
        start_idx = np.where(np.diff(ray_y_inter) == 1)[0]+1
        for i in start_idx:
            if (box_x[ray_y_inter[i]] <= ray_[0] <= box_x[ray_y_inter[i]-1] or 
                box_x[ray_y_inter[i]-1] <= ray_[0] <= box_x[ray_y_inter[i]]):
                box_idx = ray_y_inter[i]
                break
    elif len(ray_y_inter) != 0 and len(ray_x_inter) != 0: # corner
        box_idx = list(set(ray_x_inter).intersection(ray_y_inter))[0]
    return box_idx

def trace_polygon(ray1_, ray2_, same_box, world):
    box_x,box_y = same_box.coords.xy

    # find indices where each ray intersects with box
    ray1_x_inter = np.where(abs(np.array(box_x) - ray1_[0])<1e-2)[0]
    ray1_y_inter = np.where(abs(np.array(box_y) - ray1_[1])<1e-2)[0]
    ray2_x_inter = np.where(abs(np.array(box_x) - ray2_[0])<1e-2)[0]
    ray2_y_inter = np.where(abs(np.array(box_y) - ray2_[1])<1e-2)[0]
    
    box_start_idx = find_box_idx(ray1_x_inter,ray1_y_inter,ray1_,box_x,box_y)
    box_end_idx = find_box_idx(ray2_x_inter,ray2_y_inter,ray2_,box_x,box_y)
    
    print(ray1_x_inter,ray1_y_inter,ray2_x_inter,ray2_y_inter)
    print(box_start_idx,box_end_idx)
    # if box2 is not None:
    #     print(middle_start_idx,middle_end_idx)

    if box_start_idx is None or box_end_idx is None:
        return None
    else:
        if box_start_idx > box_end_idx and same_box==world:
            return np.array(same_box.coords.xy).T[box_start_idx-1:box_end_idx-1:-1]
        elif box_start_idx > box_end_idx:
            return np.hstack((
                np.array(same_box.coords.xy)[:,box_start_idx:len(same_box.coords.xy[0])],
                np.array(same_box.coords.xy)[:,1:box_end_idx])).T
        else:
            return np.array(same_box.coords.xy)[:,box_start_idx:box_end_idx].T


def find_polygon(ray_objects, world): 
    
    vs = [ray_objects[0].start,ray_objects[0].end]

    for i in range(len(ray_objects)-1):
        ray1 = ray_objects[i]
        ray2 = ray_objects[i+1]
        # print(ray1.start_box, ray1.end_box, ray2.start_box, ray2.end_box, 'box finding in utils/find_polygon')
        
        if ray1.end_box == ray2.end_box:
            if (ray1.start_box != ray2.start_box or
                ray1.start_box == ray2.start_box == LineString([[0,0],[0,0]]) or
                ray1.angle > np.pi/2 or ray2.angle< np.pi/2):
                v_bet = trace_polygon(ray1.end,ray2.end,ray1.end_box, world)
                if v_bet is not None:
                    for v in v_bet:
                        vs.append(v) if any(v != vs[-1]) else vs
                vs.append(ray2.end) if any(ray2.end != vs[-1]) else vs
                vs.append(ray2.start) if any(ray2.start != vs[-1]) else vs
                print('end=end,1',ray1.start_box != ray2.start_box,ray1.start_box == ray2.start_box == LineString([[0,0],[0,0]]),ray1.angle > np.pi/2 or ray2.angle< np.pi/2)
            else:
                if ray1.start_box == ray2.start_box:
                    vs.append(ray1.end) if all(ray1.end != vs[-1]) else vs
                    vs.append(ray1.start) if all(ray1.start != vs[-1]) else vs
                    v_bet = trace_polygon(ray1.start,ray2.start,ray1.end_box, world)
                    if v_bet is not None:
                        for v in v_bet:
                            vs.append(v) if all(v != vs[-1]) else vs
                    vs.append(ray2.start) if all(ray2.start != vs[-1]) else vs
                    vs.append(ray2.end) if all(ray2.end != vs[-1]) else vs
                    print('end=end,3, vs',vs)
                elif ray1.start_box == ray1.end_box:
                    vs.append(ray1.end) if all(ray1.end != vs[-1]) else vs
                    v_bet = trace_polygon(ray1.start,ray2.end,ray1.end_box, world)
                    if v_bet is not None:
                        for v in v_bet:
                            vs.append(v) if all(v != vs[-1]) else vs
                    vs.append(ray2.end) if all(ray2.end != vs[-1]) else vs
                    vs.append(ray2.start) if all(ray2.start != vs[-1]) else vs
                    print('end=end,2.2, vs',vs)
                elif ray2.start_box == ray2.end_box:
                    v_bet = trace_polygon(ray1.end,ray2.start,ray2.end_box, world)
                    if v_bet is not None:
                        for v in v_bet:
                            vs.append(v) if all(v != vs[-1]) else vs
                    vs.append(ray2.start) if all(ray2.start != vs[-1]) else vs
                    vs.append(ray2.end) if all(ray2.end != vs[-1]) else vs
                    print('end=end,2.1, vs',vs)

        elif ray1.start_box == ray2.start_box:
            # print('ray1 start = ray2 start')
            if ray1.start_box != LineString([[0,0],[0,0]]):
                v_bet = trace_polygon(ray1.start,ray2.start,ray1.start_box, world)
                if v_bet is not None:
                    print('start=start, v_bet',v_bet)
                    for v in v_bet:
                        vs.append(v)
                    vs.append(ray2.start)
                    vs.append(ray2.end)
            else:
                # both start at origin, but different end boxes
                v_bet = trace_polygon(ray1.end,ray2.end,ray1.end_box, world, box2 = ray2.end_box)
                if v_bet is not None:
                    print('both start at origin, but different end boxes, v_bet',v_bet)
                    for v in v_bet:
                        vs.append(v)
                    vs.append(ray2.end)
                    vs.append(ray2.start)
        elif ray1.start_box == ray2.end_box:
            v_bet = trace_polygon(ray1.start,ray2.start,ray1.end_box, world)
            if v_bet is not None:
                print('ray1 start = ray2 end, v_bet',v_bet)
                for v in v_bet:
                    vs.append(v)
                vs.append(ray2.end)
                vs.append(ray2.start)
        elif ray1.end_box == ray2.start_box:
            v_bet = trace_polygon(ray1.end,ray2.start,ray1.end_box, world)
            if v_bet is not None:
                print('ray1 end = ray2 start, v_bet',v_bet)
                for v in v_bet:
                    vs.append(v)
                vs.append(ray2.start)
                vs.append(ray2.end)
    print(vs)
    return vs