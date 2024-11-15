import math

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pqdict import pqdict
from scipy.spatial.distance import cdist
from scipy.linalg import expm
from scipy.stats import rankdata
from scipy.ndimage import median_filter

import pickle

# load model parameters
k1= 0.1;k2= 0.2
A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k1,0],[0,0,0,-k2]])
B = np.array([[0,0],[0,0],[k1,0],[0,k2]])
R = np.array([[0.5,0],[0,0.5]])

BRB = B@np.linalg.inv(R)@B.T
expA = expm(A*10**3)


class World():
    def __init__(self, 
                 map_design: np.ndarray,
                 fov: int = 60, #degrees
                 sensor_range: int = 52, #pixels
                 path_resolution: float = 0.1,
                 rr: float = 1.0,
                ):
        self.map_design = map_design
        self.map_size = map_design.shape
        self.fov = fov
        self.sensor_range = sensor_range
        self.path_resolution = path_resolution
        self.rr = rr
        self.counter = 0
    
    def update(self, new_grid: np.ndarray, state: np.ndarray) -> 'World':
        """
        Update world with new map

        Args:
            new_grid (np.ndarray): New grid observation

        Returns:
            World: Updated world
        """
        
        def loss_mask(camera_pose, fov=self.fov, grid = np.zeros((83,83))):
            # camera pose in sim frame
            mask_grid = np.ones_like(grid)
            # draw fov from camera_pose
            # convert to grid frame
            camera_pose = np.array([8-camera_pose[0], 4-camera_pose[1]])
            # convert campera pose to index
            camera_pose = (camera_pose/0.1).astype(int)

            for i in range(mask_grid.shape[0]):
                for j in range(mask_grid.shape[1]):
                    x = camera_pose[0] - i
                    y = np.abs(camera_pose[1] - j)
                    distance = np.sqrt(x**2 + y**2)
                    angle = np.arctan2(y, x)
                    angle = np.rad2deg(angle)
                    angle = (angle + 360) % 360
                    if (angle >  - fov/2) and (angle < + fov/2) and 10<distance<50:
                        mask_grid[i, j] = 0

            return mask_grid
        
        # fov mask
        mask_grid = 1 - loss_mask(self.state_to_pixel(state))

        # take intersection of occupied cells within fov 
        intersect_obs = np.logical_and(self.map_design==1, new_grid==1).astype(float)
        new_map_within_fov = np.logical_and(intersect_obs==1, mask_grid==1)

        # keep new occupied cells outside fov
        new_map_outside_fov = np.logical_and(new_grid==1, mask_grid==0).astype(float)

        # take union of free cells
        free_cells = np.where(self.map_design == 0.5)
        
        new_map = np.zeros_like(self.map_design)
        new_map[mask_grid==1] = new_map_within_fov[mask_grid==1]
        new_map[mask_grid==0] = new_map_outside_fov[mask_grid==0]
        new_map[free_cells] = 0.5
        self.map_design = new_map

        return self

    def occlusion(self, state: tuple) -> 'World':
        def bresenham_line(x0, y0, x1, y1):
            # Implementation of Bresenham's line algorithm
            # Returns the list of grid cells that a line from (x0, y0) to (x1, y1) passes through
            cells = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                cells.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy
            return cells

        def compute_occlusion(grid, observer_pos, sensor_range):
            visible_cells = set()
            x0, y0 = observer_pos
            for angle in np.arange(-self.fov/2,self.fov/2):  # Cast rays inside fov
                x1 = x0 - int(sensor_range * np.cos(angle*np.pi/180))
                y1 = y0 + int(sensor_range * np.sin(angle*np.pi/180))
                ray_cells = bresenham_line(x0, y0, x1, y1)
                
                for (x, y) in ray_cells:
                    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
                        break
                    if grid[x][y] == 1:  # 1 means occupied (obstacle)
                        break  # Stop if obstacle is found, everything behind it is occluded
                    visible_cells.add((x, y))  # Otherwise, mark as visible

            # # add a circle around the observer
            # for i in range(-3,4):
            #     for j in range(-3,4):
            #         if np.sqrt(i**2+j**2)<=3:
            #             visible_cells.add((x0+i,y0+j))

            return visible_cells
        
        visible_cells = compute_occlusion(self.map_design, state, self.sensor_range)
        pred_occ = self.map_design.copy().astype(float)
        for (x,y) in visible_cells:
            pred_occ[x][y] = 0.5
        
        # pred_occ = median_filter(pred_occ, size=2) # apply filter to cover random holes
        self.map_design = pred_occ
        # construct obstacle tree
        obstacles = np.argwhere(pred_occ != 0.5)
        self.obstacles_tree = cKDTree(obstacles)

        return self
    def state_to_pixel(self, state: np.ndarray) -> np.ndarray:
        """
        Convert planner state to pixel location

        Args:
            state (np.ndarray): State to convert

        Returns:
            np.ndarray: Pixel location
        """
        forward = state[1]
        right = state[0]
        room_size = 8

        x = int(self.map_size[0]-np.floor(forward/room_size*self.map_size[0]))
        y = int(np.floor(right/room_size*self.map_size[1]))
        pix_loc = np.array([x, y])

        return pix_loc
    
    def pixel_to_state(self, pixel: np.ndarray) -> np.ndarray:
        """
        Convert pixel location to planner state

        Args:
            pixel (np.ndarray): Pixel location

        Returns:
            np.ndarray: State
        """
        x = pixel[0]
        y = pixel[1]
        room_size = 8

        forward = (self.map_size[0]-x)/self.map_size[0]*room_size
        right = y/self.map_size[1]*room_size
        state = np.array([right, forward])

        return state

    def check_collision(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise
        """

        src = self.state_to_pixel(src)
        if dst is not None:
            dst = self.state_to_pixel(dst)


        pr = self.path_resolution
        if (dst is None) | np.all(src == dst):
            return self.obstacles_tree.query(src)[0] > self.rr

        # this needs to incorporate path/dynamics
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        yaw = math.atan2(dy, dx)
        d = math.hypot(dx, dy)
        steps = np.arange(0, d, pr).reshape(-1, 1)
        pts = src + steps * np.array([math.cos(yaw), math.sin(yaw)])
        pts = np.vstack((pts, dst))
        return bool(self.obstacles_tree.query(pts)[0].min() > self.rr)
    
    def check_collision_trajectory(self, states: np.ndarray) -> bool:
        """
        Check collision

        Args:
            states (np.ndarray): States to check

        Returns:
            bool: True if no collisions were found and False otherwise
        """
        
        for i in range(states.shape[0]):
            if self.check_collision(states[i], None)==False:
                return False
        return True

    def check_ICS(self, state: np.ndarray) -> bool:
        """
        Check collision

        Args:
            state (np.ndarray): Current state

        Returns:
            bool: True if ICS doesn't result in collision
        """
        new_state = expA@state
        return self.check_collision(new_state, None)
        

class Safe_Planner():
    def __init__(
        self,
        # map_design: np.ndarray,
        init_state: list = [3,0.5,0,0],
        n_samples: int = 1000,
        r_n: float = 20.0,
        path_resolution: float = 0.1,
        sensor_dt: float = 1,
        dt: float = 0.1,
        rr: float = 1.0,
        max_search_iter: int = 10000,
        seed: int = 0,
        Pset: np.ndarray = None,
        reachable: np.ndarray = None,
    ):
        """
        Fast Marching Tree Path Planner 

        Args:
            map_design (np.ndarray): Obstacle map described by a binary image. 1: free nodes; 0: obstacle nodes
            n_samples (int, optional): Number of nodes to sample. Defaults to 1000.
            r_n (float, optional): Range to find neighbor nodes. Defaults to .0.
            path_resolution (float, optional): Resolution of paths to check collisions. Defaults to 0.1.
            rr (float, optional): Distance threshold to check collisions. Defaults to 1.0.
            max_search_iter (int, optional): Number of maximum iterations. Defaults to 10000.
            seed (int, optional): Random seed. Defaults to 0.
        """

        # hyperparameters
        # self.map_size = map_design.shape
        self.path_resolution = path_resolution
        self.sensor_dt = sensor_dt
        self.dt = dt
        self.rr = rr
        self.n_samples = n_samples
        self.r_n = r_n
        self.max_search_iter = max_search_iter
        self.prng = np.random.RandomState(seed)  # initialize PRNG

        # construct obstacle tree
        # obstacles = np.argwhere(map_design == 1)
        # self.obstacles_tree = cKDTree(obstacles)
        self.world = World(np.ones((83,83)))
        # self.world = self.world.occlusion(self.world.state_to_pixel(init_state[0:2]))


        # initialize graph
        # SampleFree
        self.graph = nx.Graph()
        self.node_list = list()
    
        '''Load pre-computed reachable sets'''
        self.Pset = Pset
        self.reachable = reachable
        
    def goal_inter(self, start_id: int) -> np.ndarray:
        '''Returns best intermediate goal to explore'''
        start_state = self.Pset[start_id]
        v = np.sqrt(start_state[2]**2 + start_state[3]**2)

        # sample candidate goals on free space boundary
        candidates = []
        free_space = np.where(self.world.map_design == 0.5)
        # find boundary of free space
        boundary = []
        for i in range(free_space[0].shape[0]):
            left = self.world.map_design[free_space[0][i], max(0, free_space[1][i]-1)]
            right = self.world.map_design[free_space[0][i], min(self.world.map_size[1]-1, free_space[1][i]+1)]
            top = self.world.map_design[max(0, free_space[0][i]-1), free_space[1][i]]
            bottom = self.world.map_design[min(self.world.map_size[0]-1, free_space[0][i]+1), free_space[1][i]]
            if (left ==0 or right ==0 or top ==0 or bottom ==0) and (
                left !=1 and right !=1 and top !=1 and bottom !=1
            ):
                boundary.append([free_space[0][i], free_space[1][i]])

        # Sample points on the boundary evenly
        for i in range(0, len(boundary), 3):
            # convert to state
            state = self.world.pixel_to_state(np.array(boundary[i]))
            candidates.append(state)
        
        costs = []
        subgoal_ids = []

        # find the closest collision-free subgoal to each candidate
        for subgoal in candidates:
            subgoal_idx_all = np.where(rankdata(
                cdist(np.array(self.Pset)[:,0:2],subgoal.reshape(1,2)).flatten(), method='min'
                )-1==0)[0]
            valid_idx = np.where(self.bool_valid[subgoal_idx_all])[0]
            subgoal_idx = subgoal_idx_all[valid_idx]
            subgoal_ids += list(subgoal_idx)
        # compute cost to go and cost to come for each subgoal
        for subgoal_idx in subgoal_ids:
            if any([(i == subgoal_idx) for i in self.goal_explored]):
                # this subgoal is already explored
                costs.append(np.inf)
            elif subgoal_idx == start_id:
                costs.append(np.inf)
            else:
                subgoal = self.Pset[subgoal_idx]
                # cost to come
                self.goal_id = subgoal_idx
                if subgoal_idx not in self.V_unvisited:
                    cost_to_come = self.cost[self.goal_id]
                else:
                    cost_to_come = np.inf
                # cost to go
                self.goal_id = self.init_goal_id
                if subgoal[1]<=self.Pset[self.goal_id][1]:
                    dist_to_go = np.linalg.norm(np.array(subgoal[0:2])-np.array(self.Pset[self.goal_id][0:2]))
                else:
                    dist_to_go = np.inf
                if dist_to_go <= 1.0: #goal radius
                    dist_to_go = 0
                # append
                costs.append(cost_to_come + 10*dist_to_go/v)

        if all(np.isinf(costs)):
            return None, None
        else:
            self.goal_id = subgoal_ids[np.argmin(costs)]
            return self.Pset[self.goal_id]



    def plan(self,
             start: np.ndarray,
             goal: np.ndarray,
             map_design: np.ndarray) -> dict:
        """
        Run path planning

        Args:
            start (np.ndarray): Start location
            goal (np.ndarray): Goal location
            heuristic_weight (int, optional): Weight for Euclidean heuristics. Defaults to 0.0.

        Returns:
            dict:Containing path, number of steps required, and goal flag
        """

        start = np.asarray(start)
        goal = np.asarray(goal)
        # assert self.world.check_collision(start[0:2], None)
        # assert self.check_collision(goal[0:2], None)
        # update world
        self.world = self.world.update(map_design, start)
        self.world = self.world.occlusion(self.world.state_to_pixel(start[0:2]))
        
        # initialize
        self.bool_valid = np.zeros(len(self.Pset), dtype=bool)
        
        for p in range(len(self.Pset)):
            self.graph.add_node(p)
            self.node_list.append(np.array(self.Pset[p]))
            if self.world.check_collision(np.array(self.Pset[p])[0:2], None):
                self.bool_valid[p] = True
        self.V_unvisited =np.ones(len(self.Pset), dtype=bool)
        self.V_open = None
        self.time_to_come = np.zeros(self.n_samples)
        self.graph.remove_edges_from(list(self.graph.edges))

        # find nearest valid sampled node to current state and goal state
        start_idx_all = np.argsort(cdist(np.array(self.Pset),np.array([start])), axis=0)
        start_valid_idx = np.where(self.bool_valid[start_idx_all])[0]
        
        if len(start_valid_idx) == 0:
            print("Start node is in collision")
            return {
                "idx_solution": [0],
                "x_waypoints": [start],
                "u_waypoints": [[0,0]],
            }
        start_id = start_idx_all[start_valid_idx[0]][0]
        # goal doesn't have to be valid
        goal_id = np.argmin(cdist(np.array(self.Pset),np.array([goal])), axis=0)[0]
        self.init_goal_id = goal_id

        # build tree
        z, goal_flag = self.build_tree(start_id, goal_id)
        self.goal_explored = []
        if goal_flag == 1: # path found
            print("Path found")
            idx_solution = [x for x in nx.shortest_path(self.graph, start_id, z)]
        else: # no plan, explore intermediate goals
            while goal_flag == 0:
                goal_loc = self.goal_inter(start_id)
                
                if goal_loc is None or self.goal_id == goal_id:
                    goal_flag = -1
                    idx_solution = [self.goal_id]
                    print("No path found")
                    break
                else:
                    if self.goal_id not in self.V_unvisited:
                        idx_solution = [x for x in nx.shortest_path(self.graph, start_id, self.goal_id)]
                        goal_flag = 1
                        print(f"Intermediate path found from {start_id} to {self.goal_id}")
                        break # why doens't break?

        
        # output controls
        x_waypoints = []
        u_waypoints = []
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            x_waypoint, u_waypoint = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)]
            x_waypoints.append(x_waypoint)
            u_waypoints.append(u_waypoint)

        return {
            "idx_solution": idx_solution,
            "x_waypoints": x_waypoints,
            "u_waypoints": u_waypoints,
        }

    def build_tree(self, start_id: int, goal_id: int) -> None:
        """
        Build tree from a given start node

        Args:
            start_id (int): Start node id
        """
         # initialize
        goal_flag = 0
        z = start_id
        self.V_open = pqdict({z: 0.})
        self.cost = pqdict({z: 0.})
        V_closed = list()
        self.V_unvisited = list(range(len(self.node_list)))
        self.V_unvisited.remove(z)

        # start search
        for n_steps in range(self.max_search_iter):
            if z == goal_id:
                goal_flag = 1
                break
            R_plus = self.reachable[z][1][0]
            X_near = list(set(R_plus) & set(self.V_unvisited))
            
            for x in X_near:
                R_minus = self.reachable[x][2][0]
                Y_near = list(set(R_minus) & set(self.V_open))
                if len(Y_near) == 0:
                    continue
                y_min = Y_near[np.argmin([self.V_open[y] for y in Y_near])] 
                x_waypoints = self.reachable[y_min][1][3][self.reachable[y_min][1][0].index(x)][0]
                time_new = self.reachable[x][2][2][self.reachable[x][2][0].index(y_min)]
                
                # decide connection
                connect = True
                if not self.world.check_collision(self.node_list[y_min],None) or not self.world.check_collision(self.node_list[x],None):
                    connect = False
                elif self.time_to_come[y_min] + time_new <= self.sensor_dt:
                    for x_waypoint in x_waypoints[0:int(np.floor(self.sensor_dt/self.dt))]:
                        if not self.world.check_ICS(x_waypoint):
                            connect = False
                            break
                elif not self.world.check_collision_trajectory(x_waypoints): # with dynamics
                        connect = False
                    
                                
                if connect:
                    self.graph.add_edge(y_min, x)
                    self.time_to_come[y_min] = self.time_to_come[x] + time_new
                    if x in self.V_open:
                        self.V_open.updateitem(
                            x, self.V_open[y_min] + self.reachable[y_min][1][1][self.reachable[y_min][1][0].index(x)])
                        self.cost.updateitem(
                            x, self.V_open[y_min] + self.reachable[y_min][1][1][self.reachable[y_min][1][0].index(x)])
                    else:
                        self.V_open.additem(
                            x, self.V_open[y_min] + self.reachable[y_min][1][1][self.reachable[y_min][1][0].index(x)])
                        self.cost.additem(
                            x, self.V_open[y_min] + self.reachable[y_min][1][1][self.reachable[y_min][1][0].index(x)])
                    self.V_unvisited.remove(x)
            self.V_open.pop(z)
            V_closed.append(z)
            if len(self.V_open) == 0:
                # print("Search failed")
                break
            z = self.V_open.top()
        return z, goal_flag



        
