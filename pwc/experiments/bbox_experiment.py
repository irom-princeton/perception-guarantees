import numpy as np
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from pwc.experiments.base_experiment import BaseExperiment
from pwc.utils.task_util import initialize_task

class BBoxExperiment(BaseExperiment):
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
            if int(task.env) in list(range(len(task_dataset)-self.config.num_envs, len(task_dataset))):
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
        observation = env.reset(task=task)[0]
        planner.reset()
        env.dt = planner.dt # match frequency

        # initialize
        t = 0
        steps_taken = 0
        state_traj = []
        gt_obs = [[[obs[0], obs[1], obs[2]],[obs[3], obs[4], obs[5]]] for obs in task.piece_bounds_all]
        # print("GT obstacles", gt_obs)
        ground_truth = planner.world.boxes_to_planner_frame(np.array(gt_obs))
        done = False
        collided = False
        misdetected = 0
        time_misdetected = 0
        prev_policy = []
        idx_prev = 0
        plan_fail = 0

        while True and not done and not collided:
            state = planner.world.state_to_planner(env._state)
            # print(f'state at step {steps_taken}: {state}')
            boxes = perception_model.get_box(observation, experiment_config)
            # print(boxes)
            boxes[:,0,:] -= experiment_config.cp
            boxes[:,1,:] += experiment_config.cp
            boxes = planner.world.boxes_to_planner_frame(boxes)

            ###########################################################################
            X = observation[:, (observation[2, :] >0.1)]
            X = X[:, np.abs(X[1,:]) < 3.9]
            X = X[:, X[0,:]>0.05]
            X = X[:, X[0,:] < 7.95]
            
            X = np.transpose(np.array(X))
            misdetected += perception_model.count_misdetection(boxes, ground_truth, X, task.piece_bounds_all)
            # print("Misdetected: ", misdetected)
            time_misdetected+=1
            ###########################################################################

            st = time.time()
            res = planner.plan(state, boxes)
            t+=(time.time() - st)

            if (steps_taken % 10) == 0 and experiment_config.visualize:
                planner.show(res[0], true_boxes=np.array(ground_truth))
            steps_taken+=1
            if len(res[0]) > 1 and not done and not collided:
                policy_before_trans = np.vstack(res[2])
                policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
                prev_policy = np.copy(policy)

                for step in range(min(int(planner.sensor_dt/planner.dt), len(policy))):
                    idx_prev = step
                    state = env._state
                    state_traj.append(planner.world.state_to_planner(state))
                    for obs in task.piece_bounds_all:
                        if state[0] < obs[3] and state[0] > obs[0]:
                            if state[1] < obs[4] and state[1] > obs[1]: 
                                og_loc = [round(state[0]/0.1)+1 , round((state[1]+4)/0.1)+1]
                                print("Env: ", str(task.env), " Collision")
                                collided = True
                                break
                    action = policy[step]
                    obs, reward, done, info = env.step(action)
                    observation = obs[0] # get pc

                    t += planner.dt
                    if done:
                        print("Env: ", str(task.env), " Success!")
                        break
                    elif collided:
                        print("Env: ", str(task.env), " Collided")
                        break
            else:
                if (len(prev_policy) > idx_prev+1): 
                    idx_prev += 1
                    action = prev_policy[idx_prev]
                    obs, reward, done, info = env.step(action)
                    observation = obs[0] # get pc
                    t += planner.dt
                else:
                    action = [0,0]
                    obs, reward, done, info = env.step(action)
                    observation = obs[0] # get pc
                    t += planner.dt
                    plan_fail += 1
            if t > 140 or plan_fail > 10:
                print("Env: ", str(task.env), " Failed")
                break
        filename = f'{experiment_config.task.room_folder}{task.env}/cp_{experiment_config.cp}_{experiment_config.name}{experiment_config.save_tag}'
        self.plot_results(filename, state_traj , ground_truth, planner)

        result = {"trajectory": np.array(state_traj), "done": done, "collision": collided, "misdetection": (misdetected/time_misdetected)}
        np.savez_compressed(filename, data=result)
        print(f"Results saved to {filename}.npz")

        return result

    def plot_results(self, filename, state_traj , ground_truth, sp):
        fig, ax = sp.world.show(true_boxes=ground_truth)
        plt.gca().set_aspect('equal', adjustable='box')
        if len(state_traj) >0:
            state_tf = np.squeeze(np.array(state_traj)).T
            # print('state tf', state_tf.shape)
            if state_tf.shape == (4,):
                state_tf = state_tf.reshape((4,1))
            ax.plot(state_tf[0, :], state_tf[1, :], c='r', linewidth=1, label='state')
        plt.legend()
        plt.savefig(filename + f'traj_plot.png')
        # plt.savefig('plot.png')
        # plt.show()
    
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


        
