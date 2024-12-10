"""Get task dataset by aggregating room configurations.

"""

import os
import argparse
import pickle
import random


def main(args):
    # count number of non-empty folders - sometimes we generate empty folders that the task generation was aborted
    path_all = []
    for path in os.listdir(args.task_folder):
        if len(os.listdir(os.path.join(args.task_folder, path))) > 0:
            path_all += [path]
    num_tasks_available = len(path_all)
    print('Number of tasks available: {}'.format(num_tasks_available))

    # Aggregate tasks
    save_tasks = []
    for path in path_all:
        task_path = os.path.join(args.task_folder, path, 'task.pkl')
        with open(task_path, 'rb') as f:
            task = pickle.load(f)

        # Save task
        save_tasks += [task]

    # Shuffle tasks
    random.shuffle(save_tasks)

    # Save
    with open(args.save_path, 'wb') as f:
        pickle.dump(save_tasks, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path', default='/home/zm2074/Projects/data/perception-guarantees/task_1203_rot.pkl',
        nargs='?', help='path to save the task dataset'
    )
    parser.add_argument(
        '--task_folder', default='/home/zm2074/Projects/data/perception-guarantees/room_1203_rot',
        nargs='?', help='path to all the room configurations'
    )
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
