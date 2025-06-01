#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from pwc.utils.loss_fn import diff_conditional_success

#%%

Ns = np.linspace(5, 500, 100)
qs = []

for N in Ns:
    epsilon = 0.15
    delta = 0.01
    epsilon_hat = opt.bisect(diff_conditional_success, epsilon/10, 0.5, args=(1-epsilon, N, delta))
    q_level = np.ceil((N+1)*(1-epsilon_hat))/N
    qs.append(min(q_level, 1))

#%%
plt.rcParams['figure.dpi'] = 300
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(Ns, qs, '.', label='q_level')
ax.set_xlabel('Number of samples N')
ax.set_ylabel('Quantile')
ax.grid(color="#B4B4B4", linestyle='--', linewidth=0.5)
plt.savefig('num_env_vs_quantile.pdf', bbox_inches='tight')
# %%
