#%%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 300

# Example data
categories = ["Collision %", "Mis-detection %", "Goal Reached %", "Path Length (m)"]
# rotated chairs
data = {
    "NU-MCC": [0.18, 0.76, 0.82, 5.22],
    "NU-MCC-CP-avg": [0.09, 0.67, 0.90, 5.26],
    "PwC-NU-MCC": [0.00, 0.02, 0.65, 6.02],
    "PwC": [0.00, 0.00, 0.38, 7.73]
}

# # straight chairs
# data = {
#     "NU-MCC": [0.01, 0.54, 0.99, 5.32],
#     "NU-MCC-CP-avg": [0.0, 0.53, 0.99, 5.32],
#     "PwC-NU-MCC": [0.00, 0.24, 0.94, 5.68],
#     "PwC": [0.00, 0.07, 0.90, 6.45]
# }

# Bar width and spacing
bar_width = 0.5
x = np.arange(len(categories))

# Create subplots
fig, axes = plt.subplots(1, len(categories), figsize=(7,3), sharey=False)
fig.suptitle("Simulation Results for Rotated Chairs", fontsize=14)

colors = [
    (137/255, 138/255, 13/255),
    (241/255, 141/255, 0/255),
    (82/255, 177/255, 245/255),
    (32/255, 119/255, 180/255)
]

for i, (ax, category) in enumerate(zip(axes, categories)):
    for j, (method, values) in enumerate(data.items()):
        ax.bar(x[i] + j, values[i], bar_width, label=method, color=colors[j])
    
    ax.set_title(category)
    ax.set_xticks([])
    
    # Set individual y-axis limits for each subplot
    if category == "Collision %":
        ax.set_ylim(0, 0.2)
    elif category == "Mis-detection %":
        ax.set_ylim(0, 0.8)
    elif category == "Goal Reached %":
        ax.set_ylim(0.0, 0.95)
    elif category == "Path Length (m)":
        ax.set_ylim(5, 8)

# Add a single legend below the entire plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(data), fontsize=10, frameon=True, fancybox=True,edgecolor='lightgray', facecolor=(239/255, 238/255, 244/255))

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Add space for the legend
plt.savefig('experiment_results.pdf', bbox_inches='tight')

# %%
