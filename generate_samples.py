from planning.Safe_Planner import *
import ray

# sample states and compute reachability
# sp = Safe_Planner(r=10, n_samples = 1500, goal_f = [7,0,1.5,0], world_box=np.array([[0,0],[8,8]]))
# sp.find_all_reachable()

# # save pre-computed data
# f = open('planning/pre_compute/reachable-1.5k-1.5.pkl', 'ab')
# pickle.dump(sp.reachable, f)
# f = open('planning/pre_compute/Pset-1.5k-1.5.pkl', 'ab')
# pickle.dump(sp.Pset, f)


def filter_trajectory(x, u):
    if (np.all(-2<=u[:,0]) and np.all(u[:,0]<=2) 
        and np.all(-3<=u[:,1]) and np.all(u[:,1]<=3)
        # and np.all(min(vx_range)-0.4<=x[:,2]) and np.all(x[:,2]<=max(vx_range)+0.4)
        # and np.all(min(vy_range)-0.4<=x[:,3]) and np.all(x[:,3]<=max(vy_range)+0.4)
        and np.all(0.1<=x[:,0]) and np.all(x[:,0]<=7.9) and np.all(0<=x[:,1]) and np.all(x[:,1]<18)):
        return True

    return False

def filter_reachable_bound(reachable):

    reachable_new = []

    for node in reachable:
        idx = node[0]
        forward_set = node[1]
        forward_keep = []
        backward_set = node[2]
        backward_keep = []
        if len(forward_set[0]) > 0:
            for i in range(len(forward_set[0])):
                keep = filter_trajectory(forward_set[3][i][0], forward_set[3][i][1])
                forward_keep.append(keep)

        if len(backward_set[0]) > 0:
            for i in range(len(backward_set[0])):
                keep = filter_trajectory(backward_set[3][i][0], backward_set[3][i][1])
                backward_keep.append(keep)

        forward_set_new = ([forward_set[0][i] for i in range(len(forward_set[0])) if forward_keep[i]], 
                        [forward_set[1][i] for i in range(len(forward_set[1])) if forward_keep[i]], 
                        [forward_set[2][i] for i in range(len(forward_set[2])) if forward_keep[i]], 
                        [forward_set[3][i] for i in range(len(forward_set[3])) if forward_keep[i]])
        
        backward_set_new = ([backward_set[0][i] for i in range(len(backward_set[0])) if backward_keep[i]],
                            [backward_set[1][i] for i in range(len(backward_set[1])) if backward_keep[i]],
                            [backward_set[2][i] for i in range(len(backward_set[2])) if backward_keep[i]],
                            [backward_set[3][i] for i in range(len(backward_set[3])) if backward_keep[i]])
        
        reachable_new.append((idx, forward_set_new, backward_set_new))
    
    pickle.dump(reachable_new, open('planning/pre_compute/reachable_1.5_7_2K_ramp_filtered.pkl', 'wb'))
    return reachable_new

def redo_reachable(Pset, reachable):
    # change where Pset[-1] != 1.5 to 1.5
    new_Pset = Pset.copy()
    change_idx = []
    for i in range(len(Pset)-1): # except goal
        if Pset[i][-1] != 1.5:
            new_Pset[i][-1] = 1.5
            change_idx.append(i)
    # redo reachable for the changed ones

    @ray.remote
    def compute_reachable(node_idx):
        node = new_Pset[node_idx]
        print(node_idx)
        fset, fdist, ftime, ftraj = filter_reachable(node, new_Pset,1,[-2,2],[-3,3], 'F', 0.1)
        bset, bdist, btime, btraj = filter_reachable(node, new_Pset,1,[-2,2],[-3,3], 'B', 0.1)
        return (node_idx,(fset, fdist, ftime, ftraj), (bset, bdist, btime, btraj))
    
    ray.init()
    futures = [compute_reachable.remote(node_idx) for node_idx in change_idx]
    new_reachable = ray.get(futures)
    ray.shutdown()

    # update reachable
    for node in new_reachable:
        reachable[node[0]] = node
    

    return new_Pset, reachable, new_reachable

def add_ramp_up(Pset):
    close_up = np.where(np.array(Pset)[:,1] < 1)[0]


if __name__ == '__main__':
    
    reachable = pickle.load(open('planning/pre_compute/reachable_1.5_7_2K_ramp_unfiltered.pkl', 'rb'))
    Pset = pickle.load(open('planning/pre_compute/Pset_1.5_7_2K_ramp_unfiltered.pkl', 'rb'))

    new_Pset, updated_reachable, new_reachable_only = redo_reachable(Pset, reachable)

    # save new Pset and reachable
    pickle.dump(new_reachable_only, open('planning/pre_compute/reachable_1.5_7_2K_ramp_fixed_new_only.pkl', 'wb'))
    pickle.dump(new_Pset, open('planning/pre_compute/Pset_1.5_7_2K_ramp_fixed_unfiltered.pkl', 'wb'))
    pickle.dump(updated_reachable, open('planning/pre_compute/reachable_1.5_7_2K_ramp_fixed_unfiltered.pkl', 'wb'))


