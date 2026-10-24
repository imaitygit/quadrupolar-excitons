import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('HeinzGroup.xlsx', sheet_name='Fig. 2b', header=0, index_col=0)
efield = np.array([float(str(c).replace('E_hs=', '')) for c in df.columns])
energy = df.index.to_numpy(dtype=float)
Z = df.values

fig, ax = plt.subplots(figsize=(3.75, 3.26))
mask = energy <= 1.58
energy_cut = energy[mask]
Z_cut = Z[mask, :]
Z_cut = np.clip(Z_cut, 0, None)   # floor negatives to 0

# Try vmax values until it matches — start around 99.9th percentile
vmax = np.percentile(Z_cut, 99.9)
print("vmax:", vmax)

pcm = ax.pcolormesh(efield, energy_cut, Z_cut, cmap='turbo', shading='auto',
                    vmin=0, vmax=vmax)
print(np.percentile(Z, [50, 90, 95, 99, 100]))

fs = 11
ax.set_ylim(1.22, 1.58)
ax.set_yticks([1.25, 1.35, 1.45, 1.55])
# ax.set_yticklabels(['1.25', '1.35', '1.45', '1.55'], fontsize=fs)
ax.set_yticklabels([])
ax.set_xticks([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1])
ax.set_xticklabels(['-0.15', '-0.1', '-0.05', '0.0', '0.05', '0.1'], fontsize=fs)

plt.tight_layout()
print("Electric field range:", efield.min(), "to", efield.max())
plt.savefig('fig1f.png', dpi=300, pad_inches=0.1, transparent=True, bbox_inches='tight')
plt.show()
