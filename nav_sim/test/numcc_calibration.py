import pickle
from pathlib import Path
import numpy as np
import torch
import plotly.express as px
from tqdm import tqdm
base_path = Path(__file__).parent.parent.parent
data_path = base_path.parent/'data/perception-guarantees'
folder_path = data_path/'task_numcc'

import os
from nav_sim.test.numcc_generate_calibration_confidence import load_task, find_threshold
file_names = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
task_dataset = base_path.parent/'data/perception-guarantees/task_0803.pkl'

# bar = tqdm(total=len(file_names))

def process_step(step_data, task):
    pred_udf = torch.cat(step_data['all_pred_udf'],dim=0)
    cur_query_xyz = step_data['query_xyz']
    seen_xyz = step_data['seen_xyz']
    gt = step_data['gt']
    cam_position = step_data['cam_position']
    t, cov = find_threshold(pred_udf, cur_query_xyz, seen_xyz, gt, cam_position, task.piece_bounds_all)
    return t

def process_file(filename):
    with open(folder_path/filename, 'rb') as f:
        data = pickle.load(f)
        task_idx = int(filename.split('_')[-1].split('.')[0])
        task = load_task(task_dataset, task_idx)

    env_thresholds = data['thresholds']
    still_bad = []

    for step in data['bad_results'].keys():
        step_data = data['bad_results'][step]
        t = process_step(step_data, task)
        env_thresholds[step] = t
        # print(f'task {task_idx} step {step} threshold {t}')

        if t > 0.5:
            still_bad.append({'task': task_idx, 'step': step, 'threshold': t})

    # clear cuda memory
    torch.cuda.empty_cache()

    
    return max(env_thresholds), still_bad

def fix_occlusion():
    thresholds = []
    still_bad_results = []
    bar = tqdm(total=len(file_names))
    for file_name in file_names:
        bar.update(1)
        max_threshold, still_bad = process_file(file_name)
        thresholds.append(max_threshold)
        still_bad_results.extend(still_bad)

    pickle.dump(thresholds, open(base_path/'thresholds_4.pkl', 'wb'))
    pickle.dump(still_bad_results, open(base_path/'still_bad_results_4.pkl', 'wb'))

    fig = px.histogram(x=thresholds)
    # fig.show()
    fig.write_image(base_path/'thresholds_4.svg')

    return

def main():
    # this time fix near fov
    still_bad = pickle.load(open(base_path/'still_bad_results.pkl', 'rb'))
    task_dataset = base_path.parent/'data/perception-guarantees/task_0803.pkl'
    bar = tqdm(total=len(still_bad))

    still_still_bad = []
    still_bad_ts = []

    for bad in still_bad:
        bar.update(1)
        task_idx = bad['task']
        step = bad['step']
        
        task = load_task(task_dataset, task_idx)
        data = pickle.load(open(folder_path/f'task_0803_{task_idx}.pkl', 'rb'))

        t = process_step(data['bad_results'][step], task)
        still_bad_ts.append(t)

        if t > 0.5:
            print(f'task {task_idx} step {step} threshold {t}')
            still_still_bad.append({'task': task_idx, 'step': step, 'threshold': t})

    pickle.dump(still_bad_ts, open(base_path/'still_bad_ts.pkl', 'wb'))
    pickle.dump(still_still_bad, open(base_path/'still_still_bad_results.pkl', 'wb'))

if __name__ == '__main__':
    fix_occlusion()