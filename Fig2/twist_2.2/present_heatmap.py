import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from scipy.interpolate import interp1d

shift = 0.738

def get_field(f):
  m = re.search(r'sigma_(.+)\.dat', f)
  return float(m.group(1)) if m else 0.0

def load_folder(folder):
  files = sorted(glob.glob(f"{folder}/sigma_*.dat"))
  if not files:
    raise FileNotFoundError(f"No sigma_*.dat files in '{folder}'")
  fields, spectra = [], []
  for f in files:
    data = np.loadtxt(f, comments="#")
    E, sx = data[:, 0], data[:, 1] / (data[:, 0] * 18**2)
    mask = E <= (E.min() + 0.32)
    E, sx = E[mask], sx[mask]

    E_min = E.min()
    n_pad = 100
    E_pad = np.linspace(E_min - 0.4, E_min, n_pad, endpoint=False)
    sx_pad = np.zeros(n_pad)
    E  = np.concatenate([E_pad, E])
    sx = np.concatenate([sx_pad, sx])

    fields.append(get_field(f))
    spectra.append((E + shift, sx))

  pairs = sorted(zip(fields, spectra))
  return [p[0] for p in pairs], [p[1] for p in pairs]

quad_fields, quad_spectra = load_folder("quadrupolar")

def make_heatmap(fields, spectra, cmap, filename, ylabel=False):
  E_min = max(E.min() for E, _ in spectra)
  E_max = min(E.max() for E, _ in spectra)
  E_common = np.linspace(E_min, E_max, 300)

  field_conversion = 1e1  # eV/Å → V/nm
  fields_converted = np.array(fields) * field_conversion

  intensity_raw = np.zeros((len(spectra), len(E_common)))
  for i, (E, sx) in enumerate(spectra):
    interp = interp1d(E, sx, bounds_error=False, fill_value=0.0)
    sx_interp = interp(E_common)
    mx = sx_interp.max()
    intensity_raw[i] = sx_interp / mx if mx > 0 else sx_interp

  n_field = 75
  fields_uniform = np.linspace(fields_converted.min(), fields_converted.max(), n_field)
  intensity_uniform = np.zeros((n_field, len(E_common)))
  for j in range(len(E_common)):
    col_interp = interp1d(fields_converted, intensity_raw[:, j],
                          kind='cubic', bounds_error=False, fill_value=0.0)
    intensity_uniform[:, j] = col_interp(fields_uniform)

  # Transpose: energy → y-axis, field → x-axis
  img = intensity_uniform.T

  fig, ax = plt.subplots(figsize=(2.8, 3))

  extent = [
    fields_uniform[0], fields_uniform[-1],  # x: field range
    E_common[0],       E_common[-1],         # y: energy range
  ]

  ax.imshow(
    img,
    aspect="auto",
    origin="lower",
    extent=extent,
    cmap=cmap,
    vmin=0, vmax=1,
    interpolation="bicubic",
  )

  fs = 11
  ax.set_ylim(1.26, 1.58)
  ax.set_xticks([-0.12, -0.06, 0.0, 0.06, 0.12])
  ax.set_xticklabels(['-0.12', '-0.06', '0.0', '0.06', '0.12'], fontsize=fs)
  ax.set_yticks([1.35, 1.45, 1.55])
  ax.set_yticklabels(['1.35', '1.45', '1.55'], fontsize=fs)
  ax.set_yticklabels([])
  ax.set_xlim(-0.16, 0.16)
  ax.set_xlabel("Electric field (V/nm)", fontsize=fs)

  if ylabel:
    ax.set_ylabel("Energy (eV)", fontsize=fs)

  for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

  plt.tight_layout()
  plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.1)
  plt.show()
  print(f"Saved: {filename}")

make_heatmap(
  quad_fields, quad_spectra,
  plt.cm.turbo,
  "quadrupolar_heatmap_turbo.png",
  ylabel=False,
)
