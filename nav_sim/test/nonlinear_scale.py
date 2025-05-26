#%%
import torch
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import PowerTransformer

from pathlib import Path
root_path = Path(__file__).parent.parent.parent

# %%
all_pred_udf = torch.load(root_path/'all_pred_udf_0129.pth')
query_xyz = torch.load(root_path/'query_xyz_0129.pth')
all_pred_udf = torch.cat(all_pred_udf, dim=0)

# Filter out ceiling and floor
cam_position = [0.913, -3.5, 1.2]
ceiling = -2; floor = -1
cam_position_numcc = np.array([-cam_position[1], -cam_position[2], cam_position[0]])
query_xyz = query_xyz.squeeze().cpu().numpy() + cam_position_numcc

# pc to occupancy
mask = (query_xyz[:, 1] > ceiling) & (query_xyz[:, 1] < floor)
pc = query_xyz[mask]
udf = all_pred_udf.squeeze().cpu().numpy()[mask]

# project onto 2d, keep the largest udf on one pixel
grid = np.zeros((83,83))
grid_pitch = 8/82 # from get_room_from_3dfront.py
min_x = -4
min_y = 0

pc_2d = pc[:, [0, 2]]
indices = ((pc_2d - np.array([min_x, min_y])) / grid_pitch).astype(int)
if len(indices) > 0:
    indices = indices[(indices[:, 0] >= 0) & (indices[:, 0] < 83) & (indices[:, 1] >= 0) & (indices[:, 1] < 83)]    
    for i in range(len(indices)):
        x, y = indices[i]
        grid[x, y] = max(grid[x, y], udf[i].item())
# %%
# Plot the grid
fig = go.Figure()
fig.add_trace(go.Heatmap(z=grid))
fig.update_layout(title='2D Occupancy Grid', xaxis_title='X', yaxis_title='Y')
fig.show()



# %%
def make_plots(data):
    # Create a histogram
    hist_data = [data]
    group_labels = ['UDFs']
    fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=0.2, show_rug=False)
    fig_hist.update_layout(title='Histogram of All Predicted UDFs', xaxis_title='Value', yaxis_title='Density')

    # Create a Q-Q plot
    # sorted_data = np.sort(data)
    # quantiles = np.linspace(0, 1, len(sorted_data))
    # theoretical_quantiles = stats.norm.ppf(quantiles)

    # fig_qq = go.Figure()
    # fig_qq.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode='markers', name='Data'))
    # fig_qq.add_trace(go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines', name='Ideal', line=dict(dash='dash')))
    # fig_qq.update_layout(title='Q-Q Plot of All Predicted UDFs', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')

    # Show the plots
    fig_hist.show()
    # fig_qq.show()
    return

def nonlinear_transform(udf: list[torch.Tensor]) -> torch.Tensor:
    udf_numpy = udf.squeeze()
    udf_scaled_tensor = torch.exp(udf_numpy)
    return udf_scaled_tensor

def nonlinear_transform_power(udf: list[torch.Tensor]) -> torch.Tensor:
    pt = PowerTransformer(method='box-cox') # or 'yeo-johnson'
    # turn list of tensors into np array
    udf_numpy = torch.cat(udf, dim=0).squeeze().detach().cpu().numpy()
    # nonlinear transform
    udf_scaled = pt.fit_transform(udf_numpy.reshape(-1, 1))
    udf_scaled_tensor = torch.Tensor(udf_scaled.reshape(-1))
    # scale linearly back to be on [0,1]
    udf_scaled_tensor = (udf_scaled_tensor - udf_scaled_tensor.min()) / (udf_scaled_tensor.max() - udf_scaled_tensor.min())
    return udf_scaled_tensor

# %%
udf_scaled = nonlinear_transform(all_pred_udf)
make_plots(udf_scaled.cpu().numpy())


# %%
