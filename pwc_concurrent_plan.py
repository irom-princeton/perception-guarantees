from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time
import pickle
import copyreg
import copy
import threading
import random

class Robot_Plan:
    def __init__(self):
        # ****************************************************
        # SET DESIRED FLAGS
        vicon = False # set if want vicon state active; If true, make sure ./vicon.sh is running from pg_ws (and zed box connected to gas dynamics network)
        state_type = 'zed' #'zed' #if want to set the state used as the vicon state, 'zed' 
        #self.replan = True # set if want to just follow open loop plan
        #self.save_traj = False  # set if want to save trajectory and compare against plan
        #self.plot_traj = True  # set if want to visualize trajectory at the end of execution
        goal_forrestal = [7.0, -2.0, 1.5, 0.0] # goal in forrestal coordinates
        dt = 0.1 #   planner dt
        radius = 0.7 # distance between intermediate goals on the frontier
        chairs = [1, 2, 3, 4,5,6, 7, 8, 9, 10, 11, 12]  # list of chair labels to be used to get ground truth bounding boxes
        self.num_detect = 8  # number of boxes for 3DETR to detect
        # robot_radius = 0.14
        # self.cp = 0.73 # 0.73 # 0.61 #0.02
        self.cp = 0.02
        sensor_dt = 0.4 # time in seconds to replan
        num_times_detect = 1
        max_search_iter = 2000
        #result_dir = 'results/supplementary_middle_goal_dt_08_cp002/' # set to unique trial identifier if saving results
        self.result_dir = None
        self.is_finetune = False
        self.speed = 1.5
        self.scale_lateral = 0.75
        self.scale_forward = 0.8
        self.execution_steps = 10
        # ****************************************************

        # ****************************************************
        # INITIALIZE SAFE PLANNER AND GO1
        reachable_file = 'planning/pre_compute/reachable_1.5_7_2K_ramp_filtered.pkl'
        pset_file = 'planning/pre_compute/Pset_1.5_7_2K_ramp_unfiltered.pkl'
        f = open(reachable_file, 'rb')
        self.reachable = pickle.load(f)
        f = open(pset_file, 'rb')
        self.Pset = pickle.load(f)

        self.num_samples = len(self.Pset) - 1  # number of samples used for the precomputed files

        self.sp = Safe_Planner(speed=self.speed, goal_f=goal_forrestal, sensor_dt=sensor_dt, dt=dt, n_samples=self.num_samples, radius=radius, max_search_iter=max_search_iter)
        self.sp.load_reachable(self.Pset, self.reachable)
        
        self.go1 = Go1_move(self.sp.goal_f, vicon=vicon, state_type=state_type, save_folder=self.result_dir)
        # ****************************************************
        
        # ****************************************************
        # INITIALIZE POLICIES AND THREADING
        self.goal_reached = False
        self.count_used_last_plan = 1
        self.last_plan = None
        self.current_plan = None
        self.next_plan = None
        self.lock = threading.Lock()
        self.running = True
        self.boxes = []
        np.random.seed(0)
        # ****************************************************

    # ****************************************************
     # COORDINATE ADJUSTMENT FUNCTIONS
    def state_to_planner(self, state):
        # convert robot state to planner coordinates
        return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([self.sp.world.w/2,0,0,0])

    def state_to_go1(self, state):
        x, y, vx, vy = state
        return np.array([y, -x+self.sp.world.w/2, vy, -vx])

    def boxes_to_planner(self, boxes):
        boxes_new = np.zeros_like(boxes)
        for i in range(len(boxes)):
            boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([self.sp.world.w/2,0])
        return boxes_new

    def boxes_to_planner_frame(self, boxes):
        boxes_new = np.zeros_like(boxes)
        for i in range(len(boxes)):
            #boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
            boxes_new[i,0,0] = -boxes[i,1,1] + self.sp.world.w/2
            boxes_new[i,1,0] = -boxes[i,0,1] + self.sp.world.w/2
            boxes_new[i,:,1] =  boxes[i,:,0]
        return boxes_new
     # ****************************************************

    def transform_policy(self, arr):
        # print("Input Policy:", arr)
        
        # Where the input policy is empty
        if len(arr) == 0:
            # Try to resort to the last policy
            if self.last_plan is not None and len(self.last_plan[2]) > 0:
                alternative_plan = np.vstack(self.last_plan[2]) if len(self.last_plan[2]) > 1 else np.array(self.last_plan[2])
                # print("Alternative plan:", alternative_plan)

                # Determine the starting index for policy extraction
                start_index = self.execution_steps * self.count_used_last_plan
                
                # Check if the starting index exceeds the length of the alternative plan
                if start_index >= len(alternative_plan):
                    print("No more steps available in the alternative plan. Returning None.")
                    return None

                # Extract remaining steps
                policy_before_transformation = alternative_plan[start_index:]
                policy_before_transformation[:,0] *= self.scale_lateral
                policy_before_transformation[:,1] *= self.scale_forward
                self.count_used_last_plan += 1
                # print("Remaining alternative plan:", policy_before_transformation)
            else:
                print("No valid plan found. Returning None.")
                return None
            
        # Handle cases for non-empty arrays
        elif len(arr) > 1:
            # Combine multiple trajectories
            policy_before_transformation = np.vstack(arr)
            policy_before_transformation[:,0] *= self.scale_lateral
            policy_before_transformation[:,1] *= self.scale_forward

            print("Multiple trajectories in policy.")
        else:
            # Use the only trajectory in this case
            policy_before_transformation = np.array(arr[0])
            policy_before_transformation[:,0] *= self.scale_lateral
            policy_before_transformation[:,1] *= self.scale_forward

            print("Single trajectory case.")

        print("Before transformation:", policy_before_transformation)

        # Transform the policy
        transformation_matrix = np.array([[0, 1], [-1, 0]])
        policy = (transformation_matrix @ policy_before_transformation.T).T

        return policy

    def minor_shifts(self):
        movements = [
            (-0.1, 0),  # Move left
            (0.1, 0),   # Move right
            (0, -0.1)   # Move backward
        ]
        
        # Checks for free space in surroundings
        for dx, dy in movements:
            if self.sp.world.isValid([dx, dy]):
                print(f"Executing minor shift: moving by {dx} meters in x and {dy} meters in y.")
                self.go1.move((dx, dy))
                break
        else:
            print("Sampled all surrounding space, no free space found")

    def plan(self):
        print("Starting planning...")

        with self.lock:
            if self.current_plan is None or len(self.current_plan[1])==0:
                # Currently no plan in place -- must use true state
                plan_state = self.state_to_planner(self.go1.get_state()[0])
            else:
                # Combine all trajectories into one large trajectory
                combined_trajectory = np.vstack(self.current_plan[1]) if len(self.current_plan[1]) > 1 else np.array(self.current_plan[1])

                # Select the future state from the combined trajectory, up to the self.execution_steps th element (ex. 2s later) or the last element
                plan_state = combined_trajectory[min(self.execution_steps-1, len(combined_trajectory) - 1)]

        true_state = self.state_to_planner(self.go1.get_state()[0])

        print("________self proclaimed TRUE STATE: ", true_state)
        
        # Log the state from which planning will start
        print("State to plan from: ", plan_state)

        # # TODO: remove adjust for camera
        # plan_state[0][0] + 1

        # Step 1: Object detection using the camera
        t_1 = time.time()
        boxes = self.go1.camera.get_boxes(self.cp, self.num_detect, self.is_finetune,visualize=False)
        t_2 = time.time()
        print(f"Inference time: {t_2 - t_1:.4f} seconds")

        # Step 2: Extract and transform box coordinates
        boxes = boxes[:, :, 0:2]  # Extract x, y coordinates
        boxes = self.boxes_to_planner_frame(boxes)  # Transform boxes to planner's frame
        t_3 = time.time()
        print(f"Box transformation time: {t_3 - t_2:.4f} seconds")

        # Step 3: Clear explored goals and perform new planning
        self.sp.goal_explored = []
        new_plan = self.sp.plan(plan_state, boxes, true_state)
        t_4 = time.time()
        print(f"Planning time: {t_4 - t_3:.4f} seconds")
        
        if len(new_plan[0]) > 0:
            print('Saving figure...')
            fig = self.sp.show(new_plan[0], true_state=true_state, planned_state=None,true_boxes=None)
            plt.savefig(f'images/{t_4}.png', dpi=300, bbox_inches='tight')

        with self.lock:
            self.next_plan = new_plan
            print("New plan ready.")
        
        self.plan_ready.set()

    def execute(self):
        while not self.goal_reached:
            self.plan_ready.wait()

            with self.lock:
                if self.current_plan is None and self.next_plan is not None:
                    # Load the next plan into current_plan
                    self.current_plan = self.next_plan
                    self.next_plan = None
            
            if self.current_plan is not None:
                # Begin execution of the current plan
                self.executing = True
                transformed_plan = self.transform_policy(self.current_plan[2])

                # No policy to resort to
                if transformed_plan is None:
                    print("ERROR: No policy to resort to")
                    self.minor_shifts()
                    print("Replanning after minor shift...")
                # Valid policy to execute
                else:
                    num_action = 1

                    # Execute each action in the plan (up to self.execution_steps actions)
                    for action in transformed_plan[:self.execution_steps]:
                        self.go1.move(action)
                        print(f"Action number: {num_action}")

                        # Check if the goal has been reached
                        if self.go1.check_goal():
                            self.goal_reached = True
                            print("Goal reached. Stopping execution.")
                            self.executing = False
                            return

                        num_action += 1

                # Mark execution as finished and prepare for the next plan
                self.executing = False
                self.plan_ready.clear()

                # Update the last executed plan safely under lock
                with self.lock:
                    if self.current_plan is not None:
                        self.last_plan = self.current_plan
                        self.count_used_last_plan = 1
                    self.current_plan = None

    def start(self):
        # Create an Event object to synchronize planning and execution
        self.plan_ready = threading.Event()

        # Initial Plan
        self.plan()

        # Start execution of the initial plan
        execution_thread = threading.Thread(target=self.execute)
        execution_thread.start()

        # Concurrent planning during execution
        while not self.goal_reached:
            if not self.executing:
                print("Re-planning during execution...")
                self.plan()

        # Close execution thread
        execution_thread.join()


if __name__ == '__main__':
    robot_controller = Robot_Plan()
    robot_controller.start()