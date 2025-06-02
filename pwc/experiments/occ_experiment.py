import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch

from pwc.experiments.base_experiment import BaseExperiment
from pwc.utils.task_util import initialize_task

class OccExperiment(BaseExperiment):
    """
    Experiment class for planning and navigating through environments using bounding boxes.
    
    Inherits from BaseExperiment.
    """

    def __init__(self, config):
        super().__init__(config)

    def run(self,
            env,
            perception_model,
            planner,):
        task_dataset = initialize_task(self.config.task)

        for task in tqdm(task_dataset):
            if int(task.env) in [37]:# list(range(len(task_dataset)-self.config.num_envs, len(task_dataset))):
                print("Running task:", task.env)
                self.plan_env(
                    task=task,
                    env=env,
                    perception_model=perception_model,
                    planner=planner,
                    experiment_config=self.config
                )
            

    def plan_env(self, task, env, perception_model, planner, experiment_config=None):
        """
        Plan and navigate through the environment using the perception model and planner.
        
        Args:
            task: The task to be planned.
            env: The environment in which the task is executed.
            perception_model: The perception model used for planning.
            planner: The planner used to generate plans.
        """
        # reset and initialize
        observation = env.reset(task=task)
        planner.reset()
        env.dt = planner.dt # match frequency

        # ground truth
        gt_data = np.load((experiment_config.task.room_folder + str(task.env) + '/occupancy_grid.npz'), allow_pickle=True)
        gt_grid = gt_data['arr_0']
        # make gt_grid conform to planner
        map_size = planner.world.map_size
        if gt_grid.shape != map_size:
            gt = np.zeros(map_size)
            gt[:min(map_size[0],gt_grid.shape[0]), :min(map_size[1],gt_grid.shape[1])] = gt_grid[:min(map_size[0],gt_grid.shape[0]), :min(map_size[1],gt_grid.shape[1])]
            gt_grid = gt
        gt_grid = np.rot90(gt_grid, 2)

        planner_goal = planner.world.state_to_planner(np.array(self.config.goal_loc))

        # Initialize experiment variables
        t = 0
        steps_taken = 0
        state_traj = []
        done = False
        collided = False
        misdetected = 0
        time_misdetected = 0
        prev_policy = []
        idx_prev = 0
        plan_fail = 0

        
        while True and not done and not collided:
            state = planner.world.state_to_planner(env._state)
            cam_position = torch.tensor([float(env.cam_pos[0]), float(env.cam_pos[1]), float(env.cam_pos[2])])
            
            print(f'state at step {steps_taken}: {state}')
            # DETECTION
            pt = time.time()
            grid = perception_model.get_map(observation, cam_position)
            print(f'Perception time: {time.time() - pt:.2f}')

            # PLANNING
            st = time.time()
            res = planner.plan(state, planner_goal, grid)
            t+=(time.time() - st)
            print(f'Planning time: {time.time() - st:.2f}')

            steps_taken+=1

            misdetected += perception_model.count_misdetected(gt_grid, planner.world.map_design)
            time_misdetected += 1

            if experiment_config.visualize and steps_taken % 1 == 0:
                # plot grid and state with plotly
                fig = go.Figure()
                free = np.zeros((83,83))
                free[planner.world.map_design == 0.5] = 0.5
                fig.add_trace(go.Heatmap(z=gt_grid*5+planner.world.map_design))
                fig.add_trace(go.Scatter(x=[planner.world.state_to_pixel(state)[1]], y=[planner.world.state_to_pixel(state)[0]], mode='markers', marker=dict(size=10, color='red')))
                # plot plan in green
                if len(res['idx_solution']) > 1:
                    x_waypoints = np.vstack(res['x_waypoints'])
                    for i in range(len(x_waypoints)-1):
                        x1, y1 = planner.world.state_to_pixel(x_waypoints[i])
                        x2, y2 = planner.world.state_to_pixel(x_waypoints[i+1])
                        fig.add_trace(go.Scatter(x=[y1, y2], y=[x1, x2], mode='lines', line=dict(color='green', width=2)))
                fig.show()

            if len(res['idx_solution']) > 1 and not done and not collided:
                policy_before_trans = np.vstack(res['u_waypoints'])
                policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
                prev_policy = np.copy(policy)

                for step in range(min(int(planner.sensor_dt/planner.dt), len(policy))):
                    idx_prev = step
                    state = env._state
                    state_traj.append(planner.world.state_to_planner(state))
                    og_loc = planner.world.state_to_pixel(state_traj[-1])
                    if gt_grid[og_loc[0], og_loc[1]]:
                        print("Env: ", str(task.env), " Collision")
                        collided = True
                        break
                    action = policy[step]
                    observation, reward, done, info = env.step(action)
                    t += planner.dt
                    if done:
                        print("Env: ", str(task.env), " Success!")
                        break
                    elif collided:
                        print("Env: ", str(task.env), " Collided")
                        break
            else:
                plan_fail += 1
                if (len(prev_policy) > idx_prev+1): #int(planner.sensor_dt/planner.dt):
                    # for kk in range(int(planner.sensor_dt/planner.dt)):
                    idx_prev += 1
                    action = prev_policy[idx_prev]
                    state = env._state
                    state_traj.append(planner.world.state_to_planner(state))
                    observation, reward, done, info = env.step(action)
                    t += planner.dt
                else:
                    action = [0,0] # ICS was considered so shouldn't be a problem
                    state = env._state
                    state_traj.append(planner.world.state_to_planner(state))
                    observation, reward, done, info = env.step(action)
                    t += planner.dt
                    plan_fail += 1
            if t > 140 or plan_fail > 10:
                print(f"Env {task.env} Failed at t= {t} with {plan_fail} failed plans")
                break
        filename = f'{experiment_config.task.room_folder}{task.env}/cp_{experiment_config.cp}_{experiment_config.name}{experiment_config.save_tag}'
        self.plot_results(filename, state_traj , gt_grid, planner)
        # create_gif([f'{step+1}_map.png' for step in range(steps_taken-1)], 'output.gif')
        print("misdetected: ", misdetected)
        result = {"trajectory": np.array(state_traj), "done": done, "collision": collided, "misdetection": (misdetected/time_misdetected)}
        np.savez_compressed(filename, data=result)
        print(f"Results saved to {filename}.npz")

        return result

    def plot_results(self, filename, state_traj , ground_truth, sp):
        plt.clf()
        plt.imshow(ground_truth*5 + sp.world.map_design, cmap='coolwarm')
        if len(state_traj) >0:
            for state in state_traj:
                x,y = sp.world.state_to_pixel(state)[1], sp.world.state_to_pixel(state)[0]
                plt.scatter(x, y, color='red', s=1)
        plt.savefig(filename + 'traj_plot.png')
    
    def extract_results(self):
        """
        Extract results from the experiment.
        
        Args:
            task_dataset: The dataset of tasks.
            experiment_config: The configuration for the experiment.
        """
        task_dataset = initialize_task(self.config.task)
        filename = f'cp_{self.config.cp}_{self.config.name}{self.config.save_tag}'
        
        traj = {}
        done= []
        coll = []
        misdetect = []
        dist_from_goal = 0
        envs = []
        traj_length = 0

        for task in tqdm(task_dataset):
            if int(task.env) in list(range(len(task_dataset)-self.config.num_envs, len(task_dataset))):
                file_env = f'{self.config.task.room_folder}{task.env}/{filename}.npz'

                # check if file exists
                if not os.path.exists(file_env):
                    print("File does not exist: ", file_env)
                    continue

                data_ = np.load(file_env, allow_pickle=True)
                traj_info = data_["data"].item()
                traj[task.env] = traj_info['trajectory']
                envs.append(task.env)
                done.append(int(traj_info['done']))
                coll.append(int(traj_info['collision']==False))
                misdetect.append(traj_info['misdetection'])

        for i, env in enumerate(envs):
            if len(traj[env]) == 0:
                dist_from_goal += np.linalg.norm(np.array(self.config.goal_loc[0:2])-np.array(self.config.init_state[0:2]))
            else:
                if done[i] == 1:
                    traj_length+= np.sum(np.linalg.norm(np.array(traj[env][:-1,0,0:2]) - np.array(traj[env][1:,0, 0:2]), axis=1))
                if done[i] == 0:
                    dist_from_goal += np.linalg.norm(np.array(traj[env][-1,0,0:2]-np.array(self.config.goal_loc_planner_frame)))-1

        print("Average trajectory length: ", traj_length/np.sum(done))
        print("Successful task completion: ", np.mean(done))
        print("Safety rate: ", np.mean(coll))
        print("Misdetection rate: ", len(np.where(np.array(misdetect)>0)[0])/self.config.num_envs)
        print("Failed in environments: ", np.array(envs)[np.where(np.array(done)<1)[0]])
        print("Collisions in environments: ", np.array(envs)[np.where(np.array(coll)<1)[0]])
        print("Misdetections in environments: ", np.array(envs)[np.where(np.array(misdetect)>0)[0]])
        print("Average distance from goal if failed: ", dist_from_goal/(np.sum(1-np.array(done))) )


        
