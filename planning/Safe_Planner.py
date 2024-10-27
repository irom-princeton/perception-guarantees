import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pickle
import time as tm

from shapely.geometry import Point, MultiPolygon, Polygon, LineString, MultiPoint
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from shapely import contains_xy
from planning.utils import turn_box, non_det_filter, filter_reachable

# load model parameters
[k1, k2, A, B, R, BRB] = pickle.load(open('planning/sp_var.pkl','rb'))
expA = expm(A*10**3)

class World:
    def __init__(self, world_box):
        self.w = world_box[1,0] - world_box[0,0] # world width
        self.h = world_box[1,1] - world_box[0,1] # world height
        self.occ_space = None # estimated bounding boxes at this time step
        self.free_space = None # estimated free space
        self.box_space = None  # estimated occupied space

        self.counter = 0

    def update(self, new_boxes):
        '''Applies nondeterministic filter to update estimate of occupancy space'''
        new_occ_space = np.array(new_boxes)
        if self.counter == 0: # initial update
            self.box_space = unary_union(turn_box(new_boxes))
            self.box_space = self.box_space.difference(self.free_space) # make sure starts within free space
        else:
            self.box_space = non_det_filter(self.box_space, new_occ_space)

        self.occ_space = unary_union(turn_box(new_boxes)) 
        world_polygon = Polygon([[0,0],[8,0],[8,8],[0,8],[0,0]])
        self.box_space = world_polygon.intersection(self.box_space)
        
        # simplify, orient, and convert to multipolygon
        if self.occ_space.geom_type == 'Polygon':
            self.occ_space = MultiPolygon([self.occ_space])
        if self.box_space.geom_type == 'Polygon':
            self.box_space = MultiPolygon([self.box_space])

        self.occ_space.simplify(1e-5)
        self.occ_space = MultiPolygon([orient(s, sign=-1.0) for s in self.occ_space.geoms if s.geom_type == 'Polygon'])
        
        self.box_space.simplify(1e-3)
        self.box_space = MultiPolygon([orient(s, sign=-1.0) for s in self.box_space.geoms if s.geom_type == 'Polygon'])

        self.counter += 1

    def isValid(self, state):
        '''Collision check'''
        return contains_xy(self.free_space.buffer(1e-5), x=[state[0]], y=[state[1]])

    def isValid_multiple(self, states):
        '''Check collision for multiple points'''
        return np.all(contains_xy(self.free_space.buffer(1e-5), x=states[:,0:1], y=states[:,1:2]))

    def isICSfree(self, state):
        '''Check for inevitable collision set'''
        new_state = expA@state
        if self.isValid(new_state):
            return True
        else:
            return False

    def show(self, true_boxes = None):
        '''Plot free and occupied space'''
        fig, ax = plt.subplots()
        ax.set_xlim([0,self.w])
        ax.set_ylim([0,self.h])
        ax.set_aspect('equal')

        # colors for plotting
        blue = (31/255, 119/255,180/255, 0.5)
        orange = (255/255, 127/255, 14/255, 0.5)
        dark_orange = (255/255, 66/255, 15/255, 1)
        
        if self.occ_space.geom_type == 'Polygon':
            self.occ_space = MultiPolygon([self.occ_space])
        for geom in self.box_space.geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs,ys, edgecolor = dark_orange, linestyle = '--', fc=orange)
        if self.free_space is not None and self.free_space.geom_type == 'Polygon':
            xs, ys = self.free_space.exterior.xy
            ax.fill(xs,ys, edgecolor = 'k',fc=blue)
        elif self.free_space is not None and self.free_space.geom_type == 'MultiPolygon':
            for geom in self.free_space.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs,ys, edgecolor = 'k',fc=blue)
        if true_boxes is not None:
            for box in true_boxes:
                ax.add_patch(Rectangle((box[0,0],box[0,1]),box[1,0]-box[0,0],box[1,1]-box[0,1],edgecolor = 'k',linewidth = 2, fc='k'))

        return fig, ax
    
class Safe_Planner:
    def __init__(self,
                 world_box: np.ndarray = np.array([[0,0],[8,8]]), # world dimensions ([x_min, y_min], [x_max, y_max])
                 vx_range: list = [-0.5,0.5], # velocity ranges
                 vy_range: list = [0,4],
                 sr: float = 1.0, # initial clearance
                 init_state: list = [4,1,0,0.5], # initial state [x,y,vx,vy]
                 goal_f: list = [7,-2,0.5,0], # goal location with forrestal coordinates
                 dt: float = 0.1, # time resolution for controls
                 sensor_dt: float = 1, # time resolution for perception update
                 r = 3, # cost threshold for reachability
                 radius = 0.5, # radius for finding intermediate goals
                 FoV = 60*np.pi/180, # field of view
                 FoV_range = 5, # can't see beyond 5 meters
                 FoV_close = 1, # can't see within 1 meters
                 n_samples = 2000,
                 max_search_iter = 1000,
                 weight = 10,  # weight for cost to go vs. cost to come
                 seed = 0):
        
        # load inputs
        self.world_box = world_box
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.init_state = init_state
        self.world = World(world_box)
        self.goal_f = goal_f
        self.goal = self.state_to_planner(self.goal_f)

        # starts within some free space
        self.world.free_space = Polygon(((init_state[0]-sr, init_state[1]-sr),
                                         (init_state[0]-sr, init_state[1]+sr),
                                         (init_state[0]+sr, init_state[1]+sr),
                                         (init_state[0]+sr, init_state[1]-sr)))
        self.dt = dt
        self.sensor_dt = sensor_dt
        self.r = r
        self.radius = radius
        self.FoV = FoV
        self.FoV_range = FoV_range
        self.FoV_close = FoV_close
        self.n_samples = n_samples
        self.max_search_iter = max_search_iter
        self.weight = weight
        self.prng = np.random.RandomState(seed)

        # initialize
        self.cost = np.zeros(n_samples+1)
        self.time = np.zeros(n_samples+1)
        self.time_to_come = np.zeros(n_samples+1)
        self.parent = np.arange(0,n_samples+1,1, dtype=int)
        self.bool_unvisit = np.ones(n_samples+1, dtype=bool)
        self.bool_closed = np.zeros(n_samples+1, dtype=bool)
        self.bool_open = np.zeros(n_samples+1, dtype=bool)
        self.bool_valid = np.ones(n_samples+1, dtype=bool)
        self.itr = 0

    def state_to_planner(self, state):
        # convert robot state to planner coordinates
        state = np.array(state)
        state = np.atleast_2d(state)
        origin_shift = np.atleast_2d(np.array([self.world.w/2,0,0,0]))

        state_tf = (np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]]) @ state.T + origin_shift.T)
        return state_tf.squeeze()

    def find_all_reachable(self):
        '''Computes reachable sets for all pre-sampled points'''

        import ray

        # sample nodes on grid
        self.Pset = []
        num_speed = 5
        num_horizontal = int(np.ceil(np.sqrt(self.n_samples/num_speed)))
        num_vertical = int(np.ceil((self.n_samples/num_speed)/num_horizontal))
        for i in np.linspace(0.5,self.world.w-0.5,num_horizontal):
            for j in np.linspace(0.5,self.world.h-0.5,num_vertical):
                for k in np.linspace(min(self.vx_range),max(self.vx_range),num_speed):
                    self.Pset.append([i,j,k,1.5]) # constant forward speed
        self.n_samples = len(self.Pset)
        self.Pset.append(self.goal)

        # pre-compute reachable sets
        @ray.remote # speed up
        def compute_reachable(node_idx):
            node = self.Pset[node_idx]
            print(node_idx)
            fset, fdist, ftime, ftraj = filter_reachable(node,self.Pset,self.r,self.vx_range,self.vy_range, 'F', self.dt)
            bset, bdist, btime, btraj = filter_reachable(node,self.Pset,self.r,self.vx_range,self.vy_range, 'B', self.dt)
            return (node_idx,(fset, fdist, ftime, ftraj), (bset, bdist, btime, btraj))

        ray.init()
        futures = [compute_reachable.remote(node_idx) for node_idx in range(len(self.Pset))]
        self.reachable = ray.get(futures)
        ray.shutdown()

    def load_reachable(self, Pset, reachable):
        '''Load pre-computed reachable sets'''
        self.Pset = Pset
        self.reachable = reachable
        # self.point_objects = MultiPoint(np.array(self.Pset)[:,0:2])

    def goal_inter(self, start_idx):
        '''Returns best intermediate goal to explore'''
        start = self.Pset[start_idx]
        v = np.sqrt(start[2]**2 + start[3]**2)

        # sample candidate points on free space boundary
        if self.world.free_space.geom_type == 'Polygon':
            candidates = [np.array(self.world.free_space.exterior.interpolate(t).xy).reshape(2) for t in
                          np.linspace(0,self.world.free_space.length,
                                      int(np.floor(self.world.free_space.length/self.radius)),False)]
        else:
            candidates = []
            for geom in self.world.free_space.geoms:
                if geom.geom_type == 'LineString':
                    # self.world.free_space.delete(geom)
                    continue
                else:
                    candidates += [np.array(geom.exterior.interpolate(t).xy).reshape(2) for t in
                                   np.linspace(0,geom.length,
                                           int(np.floor(geom.length/self.radius)),False)]
        
        candidates = np.array(candidates)

        costs = []
        subgoal_idxs = []

        # find the closest collision-free subgoal to each candidate
        for subgoal in candidates:
            subgoal_idx_all = np.where(rankdata(
                cdist(np.array(self.Pset)[:,0:2],subgoal.reshape(1,2)).flatten(), method='min'
                )-1==0)[0]
            valid_idx = np.where(self.bool_valid[subgoal_idx_all])[0]
            subgoal_idx = subgoal_idx_all[valid_idx]
            subgoal_idxs += list(subgoal_idx)
        
        # compute cost to go and cost to come for each subgoal
        for subgoal_idx in subgoal_idxs:
            if any([all(i == self.Pset[subgoal_idx][0:2]) for i in self.goal_explored]):
                # this subgoal is already explored
                costs.append(np.inf)
            else:
                subgoal = self.Pset[subgoal_idx]
                # cost to come
                self.goal_idx = subgoal_idx
                if self.bool_unvisit[subgoal_idx]==False:
                    _, cost_to_come = self.solve(start_idx)
                else:
                    cost_to_come = np.inf
                # cost to go
                self.goal_idx = self.n_samples
                if subgoal[1]<=self.goal[1]:
                    dist_to_go = np.linalg.norm(np.array(subgoal[0:2])-np.array(self.goal[0:2]))
                else:
                    dist_to_go = np.inf
                if dist_to_go <= 1.0: #goal radius
                    dist_to_go = 0
                # append
                costs.append(cost_to_come + self.weight*dist_to_go/v)

        if all(np.isinf(costs)): # all goals explored
            return None, None
        else: # return lowest-cost subgoal
            idx_incost = np.argmin(costs)
            self.goal_idx = subgoal_idxs[idx_incost]
            return self.Pset[subgoal_idxs[idx_incost]]

    def occlusion(self, start):
        '''computes occlusion and returns free space'''

        # find all edges of bounding boxes
        edges = []
        for i in range(len(self.world.box_space.geoms)):
            b = self.world.box_space.geoms[i].exterior.coords
            edges += [LineString(b[k:k+2]) for k in range(len(b) - 1)]
        # world boundary
        world = LineString([[0,0],[0,self.world_box[1][1]],[self.world_box[1][0],self.world_box[1][1]],
                            [self.world_box[1][0],0],[0,0]])
        
        # start occlusion algorithm
        occlusion_space = self.world.box_space
        for edge in edges:
            world_intersects = []
            vertices = []
            # each edge occludes polygon beyond by line-of-sight
            for pt in edge.boundary.geoms:
                angle = np.mod(np.arctan2(pt.y-start[1],pt.x-start[0]),2*np.pi)
                ray_line = LineString([start[0:2],start[0:2]+12*np.array([np.cos(angle),np.sin(angle)])])
                world_intersection = world.intersection(ray_line)
                if world_intersection.geom_type == 'Point':
                    world_intersects.append(world_intersection)
                elif world_intersection.geom_type == 'MultiPoint':
                    world_intersects.append(world_intersection.geoms[0])
                vertices.append(pt)
            if (world_intersects[0].x != world_intersects[1].x and
                world_intersects[0].y != world_intersects[1].y):
                if world_intersects[0].x == 0 or world_intersects[1].x == 0:
                    world_intersects.append(Point([0,8]))
                elif world_intersects[0].x == 8 or world_intersects[1].x == 8:
                    world_intersects.append(Point([8,8]))
            
            # convex hull of intersections returns polygon occluded by edge
            occ = MultiPoint(vertices + world_intersects).convex_hull
            if occ.geom_type == 'Polygon':
                occlusion_space = occlusion_space.union(occ)
        
        # limited field of view
        ray_left = LineString([start[0:2],start[0:2]+12*np.array([np.cos(np.pi/2+self.FoV/2),np.sin(np.pi/2+self.FoV/2)])])
        ray_right = LineString([start[0:2],start[0:2]+12*np.array([np.cos(np.pi/2-self.FoV/2),np.sin(np.pi/2-self.FoV/2)])])
        world_intersect_left = ray_left.intersection(world)
        if world_intersect_left.geom_type == 'MultiPoint':
            world_intersect_left = world_intersect_left.geoms[0]
        fov_v = [Point(start[0:2]),world_intersect_left]
        if world_intersect_left.x != 0:
            fov_v.append(Point([0,8]))
        fov_v += [Point([0,0]),Point([8,0])]
        world_intersect_right = ray_right.intersection(world)
        if world_intersect_right.geom_type == 'MultiPoint':
            world_intersect_right = world_intersect_right.geoms[0]
        if world_intersect_right.x != 8:
            fov_v.append(Point([8,8]))
        fov_v += [world_intersect_right, Point(start[0:2])]

        occlusion_space = occlusion_space.union(Polygon(fov_v))

        world_polygon = Polygon([[0,0],[8,0],[8,8],[0,8],[0,0]])

        return world_polygon.difference(occlusion_space)

    def show(self,idx_solution, true_boxes = None):
        '''Plot solution'''
        fig, ax = self.world.show(true_boxes)
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            x_waypoints = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)][0]
            ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='red', linewidth=1)
        ax.plot(self.Pset[self.goal_idx][0],self.Pset[self.goal_idx][1],'o')
        plt.show()

    # safety planning algorithm
    def plan(self, state, new_boxes):
        '''Main function for safety planner'''

        if state.shape == (4,):
            state = state.reshape(1,4)

        # apply filter to update the world
        if self.world.occ_space is not None:
            self.world.old_occ_space = self.world.occ_space

        self.world.update(new_boxes)
        
        # occlusion
        sense_range = state[0,1]+self.FoV_range*np.cos(self.FoV/2)
        tooclose = Polygon([[state[0,0]-self.FoV_close,state[0,1]-self.FoV_close],
                            [state[0,0]+self.FoV_close,state[0,1]-self.FoV_close],
                            [state[0,0]+self.FoV_close,state[0,1]+self.FoV_close],
                            [state[0,0]-self.FoV_close,state[0,1]+self.FoV_close],
                            [state[0,0]-self.FoV_close,state[0,1]-self.FoV_close]])
        toofar = Polygon([[self.world_box[0,0],sense_range],
                                  [self.world_box[1,0],sense_range],
                                  [self.world_box[1,0],self.world_box[1,1]],
                                  [self.world_box[0,0],self.world_box[1,1]],
                                  [self.world_box[0,0],sense_range]])
        
        if contains_xy(self.world.free_space, x=[state[0,0]], y=[state[0,1]]): # currently in free space
            self.world.free_space_new = self.occlusion(state[0,:])
            if self.world.free_space_new is not None:
                self.world.free_space_new = self.world.free_space_new.difference(tooclose)
                self.world.free_space_new = self.world.free_space_new.difference(toofar)
                if self.world.free_space_new.is_valid:            
                    self.world.free_space = self.world.free_space.union(self.world.free_space_new)
                    self.world.free_space.simplify(1e-5)
        
        # initialize planner
        self.cost = np.zeros(self.n_samples+1)
        self.time = np.zeros(self.n_samples+1)
        self.time_to_come = np.zeros(self.n_samples+1)
        self.parent = np.arange(0,self.n_samples+1,1, dtype=int)
        self.bool_unvisit = np.ones(self.n_samples+1, dtype=bool)
        self.bool_closed = np.zeros(self.n_samples+1, dtype=bool)
        self.bool_open = np.zeros(self.n_samples+1, dtype=bool)
        self.bool_valid = np.zeros(self.n_samples+1, dtype=bool)
        self.itr = 0

        # check collision
        self.bool_valid = contains_xy(self.world.free_space, x=np.array(self.Pset)[:,0:1], y=np.array(self.Pset)[:,1:2]).flatten()

        # find nearest valid sampled node to current state
        start_idx_all = np.argsort(cdist(np.array(self.Pset),np.array(state)), axis=0)
        valid_idx = np.where(self.bool_valid[start_idx_all])[0]
        start_idx = start_idx_all[valid_idx[0]][0]
        self.goal_idx = self.n_samples
        self.bool_open[start_idx] = True

        # solve
        goal_flag = self.build_tree(start_idx)
        self.goal_explored = []
        if goal_flag == 1: # plan found
            idx_solution, _ = self.solve(start_idx) 
        else: # no plan, explore intermediate goals
            while goal_flag == 0:
                goal_loc = self.goal_inter(start_idx)
                if goal_loc is None or self.goal_idx == self.n_samples:
                    goal_flag = -1
                    idx_solution = [self.goal_idx]
                    break
                else:
                    self.goal_explored.append(self.Pset[self.goal_idx][0:2])
                    if self.bool_unvisit[self.goal_idx]==False:
                        idx_solution, _ = self.solve(start_idx)
                        break

        # output controls
        x_waypoints = []
        u_waypoints = []
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            x_waypoint, u_waypoint = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)]
            x_waypoints.append(x_waypoint)
            u_waypoints.append(u_waypoint)

        return idx_solution, x_waypoints, u_waypoints

    def build_tree(self, start_idx):
        '''Builds tree for FMT*'''
        self.bool_open[start_idx] = True
        goal_flag = 0
        while self.itr <= self.max_search_iter:
            self = self.extend()
            if not self.bool_unvisit[self.goal_idx]: # goal node is visited
                goal_flag = 1
                break
        self.itr=1
        return goal_flag

    def solve(self, start_idx):
        '''Main FMT* algorithm'''
        idx = self.goal_idx
        idx_solution = [idx]
        while True:
            idx = self.parent[idx]
            idx_solution.append(idx)
            if idx == start_idx: # start node
                break

        tot_cost = self.cost[self.goal_idx]

        return idx_solution[::-1], tot_cost

    def extend(self):
        '''Inner loop of FMT*'''
        self.itr += 1

        # check nodes are collision-free
        self.idxset_open = np.where(self.bool_open & self.bool_valid)[0] # H
        idxset_unvisit = np.where(self.bool_unvisit & self.bool_valid)[0] # W
        if self.idxset_open.size != 0:
            idx_lowest = self.idxset_open[np.argmin(self.cost[self.idxset_open])] # z <- argmin cost(y)

            R_plus = self.reachable[idx_lowest][1][0]
            idxset_near = list(set(R_plus) & set(idxset_unvisit)) # X_near <- R+(z) \cap W
            # for x in X_near
            for idx_near in idxset_near:
                self.process_idx(idx_near)
                
            self.bool_open[idx_lowest] = False
            self.bool_closed[idx_lowest] = True

        return self

    def process_idx(self, idx_near):
        '''Processes a node in FMT*'''
        R_minus = self.reachable[idx_near][2]
        idxset_cand = list(set(R_minus[0]) & set(self.idxset_open)) #index in Pset
        idxset_inR = np.where(np.isin(R_minus[0], idxset_cand))[0]
        distset_cand = np.array(R_minus[1])[idxset_inR]
        timeset_cand = np.array(R_minus[2])[idxset_inR]

        if len(idxset_cand) == 0:
            pass
        else:
            # ymin <- argmin cost(y) + dist(y,x)
            idx_incand_costmin = np.argmin(self.cost[idxset_cand] + distset_cand) #index in cand set
            cost_new = min(self.cost[idxset_cand] + distset_cand)
            time_new = timeset_cand[idx_incand_costmin]
            idx_parent = idxset_cand[idx_incand_costmin]
            idx_nearinparentfset = self.reachable[idx_parent][1][0].index(idx_near)
            x_waypoints = self.reachable[idx_parent][1][3][idx_nearinparentfset][0]
            connect = True
            if self.bool_valid[idx_near] == False or self.bool_valid[idx_parent] == False:
                connect = False
            elif not self.world.isValid_multiple(x_waypoints):
                connect = False
            elif self.time_to_come[idx_parent] + time_new <= self.sensor_dt:
                for x_waypoint in x_waypoints[0:int(np.floor(self.sensor_dt/self.dt))]:
                    if not self.world.isICSfree(x_waypoint):
                        connect = False
                        break
            
            if connect:
                self.connect(idx_near,cost_new,time_new,idx_parent)

    def connect(self,idx_near, cost_new, time_new, idx_parent):
        '''Connects nodes in FMT*'''
        self.bool_unvisit[idx_near] = False
        self.bool_open[idx_near] = True
        self.cost[idx_near] = cost_new
        self.time[idx_near] = time_new
        self.parent[idx_near] = idx_parent
        self.time_to_come[idx_near] = self.time_to_come[idx_parent] + time_new


    def plot_reachable(self, direction):
        '''Plot reachability connections'''
        _, ax = plt.subplots()
        ax.set_xlim([0,self.world.w])
        ax.set_ylim([0,self.world.h])
        for i in range(self.n_samples):
            ax.scatter(self.Pset[i][0],self.Pset[i][1],color = 'k',marker = '.')
            if direction == 'F':
                fset = self.reachable[i][1]
                for j in range(len(fset[0])):
                    # show_trajectory(ax, self.Pset[i],
                    #                 self.Pset[fset[0][j]],fset[1][j],self.dt)
                    traj = self.reachable[i][1][3][j][0]
                    ax.plot(traj[:,0], traj[:,1], c='gray', linewidth=0.5)

            elif direction == 'B':
                bset = self.reachable[i][2]
                for j in range(len(bset[0])):
                    # show_trajectory(ax, self.Pset[bset[0][j]],
                    #                 self.Pset[i],bset[1][j],self.dt)
                    traj = self.reachable[i][2][3][j][0]
                    ax.plot(traj[:,0], traj[:,1], c='gray', linewidth=0.5)
        plt.show()