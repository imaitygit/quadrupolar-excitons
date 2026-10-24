import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

# Shift energies to align the zero-field quad peak to 1.4 eV.
# Rigid shift
shift = 0.69
# quadrupolar = 0.69; dipolar = 0.72
peak_window = 0.068

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
        E, sx = data[:, 0], data[:, 1] / (data[:, 0] * 6**2)
        mask = E <= (E.min() + 0.4)
        E, sx = E[mask], sx[mask]
        # Pad zeros below E.min() down to E.min() - 0.4
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

# Quadrupolar and dipolar data
quad_fields, quad_spectra = load_folder("quadrupolar")
dip_fields,  dip_spectra  = load_folder("dipolar")

def make_heatmap(fields, spectra, cmap, filename, ylabel=True):
    # Build a common energy grid
    E_min = max(E.min() for E, _ in spectra)
    E_max = min(E.max() for E, _ in spectra)
    E_common = np.linspace(E_min, E_max, 300)

    # V/nm conversion (no mV scaling)
    field_conversion = 1e1                          # <-- changed from 1e4
    fields_converted = np.array(fields) * field_conversion

    # Normalise each spectrum to its own peak
    intensity_raw = np.zeros((len(spectra), len(E_common)))
    for i, (E, sx) in enumerate(spectra):
        interp = interp1d(E, sx, bounds_error=False, fill_value=0.0)
        sx_interp = interp(E_common)
        mx = sx_interp.max()
        intensity_raw[i] = sx_interp / mx if mx > 0 else sx_interp

    # Interpolate onto a uniform field grid
    n_field = 75
    fields_uniform = np.linspace(fields_converted.min(), fields_converted.max(), n_field)
    intensity_uniform = np.zeros((n_field, len(E_common)))
    for j in range(len(E_common)):
        col_interp = interp1d(fields_converted, intensity_raw[:, j],
                              kind='cubic', bounds_error=False, fill_value=0.0)
        intensity_uniform[:, j] = col_interp(fields_uniform)

    # --- Flip: transpose so energy → y-axis, field → x-axis ---
    # intensity_uniform shape: (n_field, n_energy)  →  transpose → (n_energy, n_field)
    img = intensity_uniform.T

    # fig, ax = plt.subplots(figsize=(3.75, 3))
    if prefix == "dipolar":
        fig, ax = plt.subplots(figsize=(4.2, 3.24))
    else:
        fig, ax = plt.subplots(figsize=(4.2, 3.26))

    extent = [
        fields_uniform[0],  fields_uniform[-1],   # x: field range
        E_common[0],        E_common[-1],          # y: energy range
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

    fs = 12
    ax.set_ylim(1.22, 1.58) 
    ax.set_xticks([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_yticks([1.25, 1.35, 1.45, 1.55])
    if prefix == "dipolar":
    # ax.set_xlabel("Electric field (V/nm)", fontsize=fs)   # <-- now on x 
        # ax.set_yticklabels([])                              # energy limits on y
        ax.set_xticklabels([])  # field ticks on x
    else:
        ax.set_xticklabels(['-0.15', '-0.1', '-0.05', '0.0', '0.05', '0.1'], fontsize=fs)
    ax.set_xlim(-0.1896, 0.1354)
    ax.set_yticklabels(['1.25', '1.35', '1.45', '1.55'], fontsize=fs)

    if ylabel:
        ax.set_ylabel("Energy (eV)", fontsize=fs)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.1)
    plt.show()
    print(f"Saved: {filename}")

# Plasma colormap only
for fields, spectra, prefix, yl in [
    #(dip_fields,  dip_spectra,  "dipolar",      False),
    (quad_fields, quad_spectra, "quadrupolar",  False),
]:
    make_heatmap(fields, spectra, plt.cm.turbo, f"{prefix}_heatmap_turbo.png", ylabel=yl)
