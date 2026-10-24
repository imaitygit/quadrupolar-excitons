import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# ── configuration ──────────────────────────────────────────────────────────────
stackings = ["BA", "AB", "AA"]

hole_positions = {
  "BA": {
    "l3": np.array([60.2, 34.8, 16.1]),
    "l2": np.array([61.3, 33.3, 10.1]),
    "l1": np.array([60.2, 24.8,  3.8]),
  },
  "AB": {
    "l3": np.array([26.1, 15.7, 16.5]),
    "l2": np.array([25.5, 14.0, 10.1]),
    "l1": np.array([26.1, 15.7,  3.8]),
  },
  "AA": {
    "l3": np.array([2.4, 2.3, 16.9]),
    "l2": np.array([2.5, 1.9, 10.1]),
    "l1": np.array([2.4, 2.3,  3.4]),
  },
}

hole_colors = {
  "AA": "red",
  "AB": "blue",
  "BA": "pink",
}

def group_z_into_layers(z_values, gap_threshold=2.0):
  """Cluster z-values into physical layers by detecting large gaps."""
  zs = np.unique(np.round(z_values, 3))
  gaps = np.diff(zs)
  split_indices = np.where(gaps > gap_threshold)[0] + 1
  clusters = np.split(zs, split_indices)
  layer_z = np.array([c.mean() for c in clusters])
  print(f"  Found {len(clusters)} layer(s) from {len(zs)} unique z-values:")
  for i, (c, lz) in enumerate(zip(clusters, layer_z)):
    print(f"    Layer {i+1}: z in {c} → unit cell at z={lz:.3f} Å")
  return layer_z

# ── static data loaded once ────────────────────────────────────────────────────
Rvec = np.loadtxt("Rvec.txt")
wf   = np.loadtxt("pos_wf.txt")

num_orb     = wf[:, 2] - wf[:, 1]
selected_wf = wf[np.isclose(num_orb, 5.0, atol=1e-5)]

layer_z_positions = group_z_into_layers(selected_wf[:, 5], gap_threshold=2.0)

a       = 55.8751612617
a1      = np.array([a, 0])
a2      = np.array([a * 0.5, a * np.sin(np.radians(60))])
origin  = np.array([0, 0])
cell_xy = np.array([origin, a1, a1 + a2, a2, origin])

vmin = 0.0
# Peak I: (6, 7)
#vmax = 7.333e-04
# Peak II: (24, 25)
vmax = 1.555e-04 

# ── visual knobs ───────────────────────────────────────────────────────────────
GRID_RES    = 300
SIGMA_A     = 2.0    # Gaussian spread in Å
ALPHA_GAMMA = 0.5
ALPHA_MAX   = 0.90
ATOM_SIZE   = 0.5
ATOM_ALPHA  = 0.50
ATOM_COLOR  = 'black'

cmap_obj = LinearSegmentedColormap.from_list(
  'white_to_vivid', ['white', '#FF4000'], N=512
)

# ── grid ───────────────────────────────────────────────────────────────────────
cx, cy  = 55.0, 0.0
half_xy = 110.0
xs_grid = np.linspace(cx - half_xy, cx + half_xy, GRID_RES)
ys_grid = np.linspace(cy - half_xy, cy + half_xy, GRID_RES)

# ── smeared-space equivalent of vmax ──────────────────────────────────────────
# Gaussian smearing dilutes a single atom of weight w into a peak of
# w / (2π × sigma_px²).  d_vmax is the smeared density that corresponds
# to one atom sitting exactly at vmax — our normalisation ceiling.
dx       = (xs_grid[-1] - xs_grid[0]) / (GRID_RES - 1)
dy       = (ys_grid[-1] - ys_grid[0]) / (GRID_RES - 1)
sigma_px = 0.5 * (SIGMA_A / dx + SIGMA_A / dy)
d_vmax   = vmax / (2.0 * np.pi * sigma_px ** 2)
print(f"d_vmax (smeared-space ceiling) = {d_vmax:.4e}")

# ── helpers ────────────────────────────────────────────────────────────────────

def gaussian_smear_2d(xy, weights, xs, ys, sigma_a):
  nx, ny   = len(xs), len(ys)
  dx_      = (xs[-1] - xs[0]) / (nx - 1)
  dy_      = (ys[-1] - ys[0]) / (ny - 1)
  density, _, _ = np.histogram2d(
    xy[:, 0], xy[:, 1],
    bins=[nx, ny],
    range=[[xs[0], xs[-1]], [ys[0], ys[-1]]],
    weights=weights,
  )
  density  = density.T
  sigma_px_ = 0.5 * (sigma_a / dx_ + sigma_a / dy_)
  density  = gaussian_filter(density, sigma=sigma_px_)
  return density


def render_density_layer(ax, d_norm, z_layer, xs, ys,
                         cmap, alpha_gamma, alpha_max):
  X, Y = np.meshgrid(xs, ys)
  Z    = np.full_like(X, z_layer)
  rgba         = cmap(d_norm)
  rgba[..., 3] = np.clip(d_norm ** alpha_gamma, 0.0, alpha_max)
  ax.plot_surface(
    X, Y, Z,
    facecolors  = rgba,
    shade       = False,
    antialiased = False,
    rcount      = len(ys),
    ccount      = len(xs),
    linewidth   = 0,
  )

# ── main loop ──────────────────────────────────────────────────────────────────
layer_key = "l1"
#s_list    = [6, 7]
s_list    = [24, 25]
s_label   = "_".join(map(str, s_list))

for stacking in stackings:
  point              = hole_positions[stacking][layer_key]
  current_hole_color = hole_colors.get(stacking, "red")

  filename_hdf5 = f"Exciton_rh_{stacking}_{layer_key}_tb.hdf5"
  print(f"\n{'='*60}")
  print(f"Processing: {filename_hdf5}")
  print(f"  S={s_list} | Hole: {current_hole_color}")

  with h5py.File(filename_hdf5, "r") as f:
    exciton_wfn = f["exciton_wfn"][:]

  positions, weights = [], []
  for j in range(len(selected_wf)):
    offset    = selected_wf[j, 3:]
    shifted_R = Rvec + offset
    positions.append(shifted_R)
    avg_intensity = np.mean(np.abs(exciton_wfn[s_list, j, :])**2, axis=0)
    weights.append(avg_intensity)

  pos         = np.vstack(positions)
  weights_arr = np.hstack(weights)
  print(f"  Max: {weights_arr.max():.3e}  Min: {weights_arr.min():.3e}")

  layer_idx = np.argmin(
    np.abs(pos[:, 2:3] - layer_z_positions[np.newaxis, :]), axis=1
  )

  fig = plt.figure(figsize=(5, 5))
  ax  = fig.add_subplot(111, projection='3d')
  ax.set_axis_off()
  ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
  ax.grid(False)

  for li, z_layer in enumerate(layer_z_positions):
    mask = layer_idx == li
    xy_l = pos[mask, :2]
    wt_l = weights_arr[mask]

    density = gaussian_smear_2d(xy_l, wt_l, xs_grid, ys_grid, SIGMA_A)

    # vmax → d_vmax → d_norm = 1.0 → full red
    # vmin = 0 → d_norm = 0.0 → white
    d_norm = np.clip(density / d_vmax, 0.0, 1.0)

    render_density_layer(
      ax, d_norm, z_layer,
      xs_grid, ys_grid,
      cmap_obj,
      alpha_gamma = ALPHA_GAMMA,
      alpha_max   = ALPHA_MAX,
    )

    ax.scatter(
      xy_l[:, 0], xy_l[:, 1],
      zs         = z_layer,
      zdir       = 'z',
      c          = ATOM_COLOR,
      s          = ATOM_SIZE,
      alpha      = ATOM_ALPHA,
      edgecolors = 'none',
      depthshade = False,
      rasterized = True,
    )

  for z_layer in layer_z_positions:
    ax.plot(
      cell_xy[:, 0], cell_xy[:, 1],
      zs=z_layer, zdir='z',
      color='grey', alpha=0.7, lw=1.5, ls="-",
    )

  # ax.scatter(*point, color=current_hole_color, marker='o', s=35,
  #            linewidth=0.8, zorder=20)

  x_range = pos[:, 0].max() - pos[:, 0].min()
  y_range = pos[:, 1].max() - pos[:, 1].min()
  ax.set_box_aspect((x_range, y_range, x_range * 0.75))
  ax.set_xlim(cx - half_xy, cx + half_xy)
  ax.set_ylim(cy - half_xy, cy + half_xy)

  ax.view_init(elev=45, azim=-81, roll=0)

  out_name = f"I_Q_3D_{stacking}_{layer_key}_S{s_label}_avg.png"
  plt.savefig(out_name, dpi=300, bbox_inches='tight',
              transparent=True, pad_inches=0.1)
  plt.show()
  plt.close()
  print(f"  Saved {out_name}")
