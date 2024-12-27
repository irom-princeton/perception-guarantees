import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool
import math
import IPython as ipy
import plotly.graph_objects as go
from itertools import product, combinations


from sklearn.cluster import DBSCAN

def is_box_visible(X, obstacles, visualize):
    # t_s = time.time()
    is_vis = [False]*len(obstacles)
    noise = np.array(X.T)
    num_points=len(noise)
    for obs_idx, obs in enumerate(obstacles):
        # Check if any visible points are in the ground truth boxes. If more than 100 points are inside the box, it is marked visible
        if (num_points > 0):
            obs = np.array(obs)
            s=[(noise[:,i]>obs[i]+0.1) & (noise[:,i]<obs[3+i]-0.1) for i in range(3)]
            s=np.array(s)
            is_vis_noise=bool(sum(s[0,:]&s[1,:]&s[2,:])>100)
        else: 
            is_vis_noise = False
        is_vis[obs_idx]  = is_vis_noise
    # t_e = time.time()
    # print("time for checking what's visible ", t_e-t_s)
    if visualize:
        # Visualize
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=noise[:,0], 
            y=noise[:,1], 
            z=noise[:,2],
            mode='markers',
            marker=dict(size=1)
        ))
        gt_obs = [[[obs[0], obs[1], obs[2]],[obs[3], obs[4], obs[5]]] for obs in obstacles]
        gt_boxes = np.array(gt_obs)

        for jj, cc in enumerate(gt_boxes):
            r0 = [cc[0, 0], cc[1, 0]]
            r1 = [cc[0, 1], cc[1, 1]]
            r2 = [cc[0, 2], cc[1, 2]]

            for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                    np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                    np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                    fig.add_trace(go.Scatter3d(
                        x=[s[0], e[0]], 
                        y=[s[1], e[1]], 
                        z=[s[2], e[2]],
                        mode='lines',
                        line=dict(color='green')
                    ))

        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        fig.show()
    return is_vis

def cluster(X, visualize):
    db = DBSCAN(eps=0.5, min_samples=100).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    cluster_centers = np.zeros((n_clusters_, 3))

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        if(len(xy) > 0):
            cluster_centers[k, :] = np.mean(xy, axis=0)
            
        if visualize:
            # plt.plot(
            #     xy[:, 0],
            #     xy[:, 1],
            #     "o",
            #     markerfacecolor=tuple(col),
            #     markeredgecolor="k",
            #     markersize=14,
            # )
            ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                marker='o',
                color=col
            )

            xy = X[class_member_mask & ~core_samples_mask]
            ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                marker='^',
                color=col
            )
            # plt.show()
    if visualize:
        ax.set_aspect('equal')
        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()
    return cluster_centers