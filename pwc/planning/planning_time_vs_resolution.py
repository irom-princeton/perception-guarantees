#%%
import matplotlib.pyplot as plt
import numpy as np

# Define planning times
planning_times = {
    120: [4.21, 4.33, 4.64, 4.61, 4.72, 4.60, 4.69, 4.82, 5.01, 4.20,
          3.92, 3.49, 3.30, 3.07, 2.74, 2.43, 2.10, 1.82, 1.64, 1.47,
          1.21, 1.09, 0.85, 0.86, 0.65, 0.65, 0.47, 0.33, 0.29, 0.19,
          0.19, 1.08, 1.09, 1.12, 1.11],
    100: [4.09, 4.22, 4.19, 4.14, 4.36, 4.54, 4.67, 4.87, 4.34, 4.12,
          3.56, 3.48, 3.34, 3.14, 2.81, 2.41, 2.10, 1.79, 1.65, 1.51,
          1.13, 0.96, 0.71, 0.72, 0.48, 0.36, 0.27, 0.27, 0.17, 0.17,
          0.98, 1.01, 1.02, 1.03],
    80: [4.32, 4.38, 4.38, 4.38, 4.51, 4.59, 4.73, 4.93, 5.00, 3.93,
         3.57, 3.33, 3.10, 2.83, 2.48, 2.12, 1.81, 1.65, 1.61, 1.40,
         1.18, 0.95, 0.69, 0.70, 0.47, 0.35, 0.25, 0.26, 0.15, 0.15,
         0.92, 0.94, 0.95, 0.98],
    60: [4.52, 4.51, 4.49, 4.41, 4.44, 4.54, 4.82, 4.88, 4.48, 3.88,
         3.33, 3.20, 2.99, 2.70, 2.39, 2.08, 1.75, 1.61, 1.57, 1.36,
         1.16, 0.93, 0.68, 0.69, 0.45, 0.33, 0.24, 0.24, 0.14, 0.15,
         0.83, 0.81, 0.84, 0.84],
    40: [3.90, 4.04, 3.89, 3.90, 4.36, 4.32, 4.28, 4.50, 4.67, 4.17,
         3.89, 3.39, 3.17, 3.18, 2.39, 2.08, 1.76, 1.61, 1.56, 1.36,
         1.21, 0.79, 0.79, 0.59, 0.59, 0.40, 0.27, 0.23, 0.14, 0.14,
         0.67, 0.69, 0.69, 0.71]
}

# Compute mean and std
resolutions = sorted(planning_times.keys())
means = [np.mean(planning_times[res]) for res in resolutions]
stds = [np.std(planning_times[res]) for res in resolutions]

#%% PLot
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(resolutions, means, yerr=stds, fmt='o-', capsize=5, label='Planning Time')
ax.set_xlabel('Map Resolution (pixels)')
ax.set_ylabel('Planning Time (seconds)')
ax.grid(color="#B4B4B4", linestyle='--', linewidth=0.5)
plt.savefig('planning_time_vs_resolution.pdf', bbox_inches='tight')
# %%
