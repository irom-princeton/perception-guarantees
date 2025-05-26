from planning.Safe_Planner import *

def state_to_planner(state, sp):
    # convert robot state to planner coordinates
    return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([sp.world.w/2,0,0,0])

def state_to_go1(state, sp):
    x, y, vx, vy = state
    return np.array([y, -x+sp.world.w/2, vy, -vx])

def boxes_to_planner(boxes, sp):
    boxes_new = np.zeros_like(boxes)
    for i in range(len(boxes)):
        boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
    return boxes_new

def get_boxes(true_boxes, cp):
    # fake random boxes in planner coordinates
    # replace with camera + 3detr later
    # n = np.random.randint(1,5)
    boxes = []
    n = len(true_boxes)
    for i in range(n):
        x0 = np.random.uniform(cp/3,cp)
        y0 = np.random.uniform(cp/3,cp)
        x1 = np.random.uniform(cp/3,cp)
        y1 = np.random.uniform(cp/3,cp)
        boxes.append(np.array([[true_boxes[i,0,0]-x0,
                                true_boxes[i,0,1]-y0],
                               [true_boxes[i,1,0]+x1,
                                true_boxes[i,1,1]+y1]]))
    return np.array(boxes)



def plan_loop():

    # planner
    # load pre-computed: need to recompute for actual gains
    f = open('planning/pre_compute/reachable-2k.pkl', 'rb')
    reachable = pickle.load(f)
    f = open('planning/pre_compute/Pset-2k.pkl', 'rb')
    Pset = pickle.load(f)

    # print(Pset[-1])

    # initialize planner
    init_state = [5,0.2,0,0]
    sp = Safe_Planner(goal_f=[7,0,1.5,0],
                      sr = 0.01, 
                      FoV_close=0,
                      init_state=init_state,
                      radius = 1,
                      n_samples=len(Pset)-1,
                      world_box=np.array([[0,0],[8,8]]), 
                      max_search_iter=1000)
    sp.load_reachable(Pset, reachable)
    # print(Pset[0:20])
    cp = 0.6
    true_boxes = np.array([[[1,4],[3.5,6]],
                           [[2,3],[2.5,3.5]],
                           [[5.3,2.5],[6,3]]],
                           )
    # true_boxes = np.array([[[1,4],[1.5,4.5]]])
    boxes = get_boxes(true_boxes,cp)
    # boxes = true_boxes.copy()
    # boxes[:,0,:] -= cp
    # boxes[:,1,:] += cp
    # boxes = np.array([[[2,3],[5,7]]])
    # # boxes = np.array([[[0,0],[0.1,0.1]]])
    # plan
    # boxes = true_boxes
    # boxes[:,0,:] -= cp
    # boxes[:,1,:] += cp
    state = np.array([init_state])
    res = sp.plan(state, boxes)
    #plt.figure(1)
    # sp.world.show(true_boxes)
    
    # sp.show(res[0], true_boxes = true_boxes)
    print(res[0])
    iter = 0
    while True:
        # fig = sp.world.show(true_boxes)
        fig = sp.show(res[0],true_boxes)
        # save figure
        plt.savefig('planning/plots/{}.png'.format(iter))
        if np.linalg.norm(state-sp.goal) <= 1:
            print('yay')
            break

        state = np.array([res[1][0][-1]])
        boxes = get_boxes(true_boxes,cp)
        print(state)
        # boxes = true_boxes.copy()
        # boxes[:,0,:] -= cp
        # boxes[:,1,:] += cp
        # sp.world.update(boxes)
        # fig, ax = sp.world.show()
        # plt.show()
        start = tm.time()
        res = sp.plan(state, boxes)
        print('time:', tm.time()-start)
        
        
        print(res[0])
        iter += 1

        



if __name__ == '__main__':
    plan_loop()
    