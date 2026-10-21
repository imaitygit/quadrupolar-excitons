"""
Local strain map between two LAMMPS data files (unrelaxed vs relaxed).

Atoms of the chosen type are paired by atom ID.  The structure is then
replicated NX x NY times in plane, so the moire pattern is visible over more
than one period and bonds run continuously across the old cell edges.  The
neighbour list is built once in the unrelaxed structure, using the minimum
image convention of the enlarged cell, and the same pairs are followed into
the relaxed structure.  Each atom's strain is the average length change of its
bonds,

    strain_i = < (d_relaxed - d_unrelaxed) / d_unrelaxed >

which does not change under rotation, so a local twist does not show up as
strain.  Bonds that wrap around the outer periodic boundary are used in the
strain but not drawn, so no long lines cross the figure.

Two boundary details matter once NX, NY > 1:

  * An atom that crosses the cell edge during relaxation is written back on
    the far side of the box, a whole lattice vector from where it started.
    In a single cell the minimum image folds that away.  In the tiled cell the
    minimum image only removes multiples of NX * a1, so the jump survives and
    shows up as a bond stretched by a full lattice vector.  Each relaxed atom
    is therefore folded onto the image nearest its unrelaxed partner first.

  * Both structures are then folded into the cell by the same integer lattice
    shift, taken from the unrelaxed one, so the fold above is not undone and
    pairs sitting on the boundary are not split between tiles.

Note on cost: the neighbour search is O(N^2) per periodic image, so NX x NY
tiles cost (NX * NY)^2 in time and memory.  3 x 3 is 81x the single cell.

Output: strain_map_<type>.png

Run:  python simple_strain_map.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ----------------------------- CONFIG ------------------------------------

UNRELAXED = "lammps.dat.unrelaxed"
RELAXED = "lammps.dat.relaxed"
ATOM_TYPE = 7          # 1, 4 or 7 for W of each layer
CUTOFF = 4.3           # Angstrom, between the first and second W-W shell
DRAW_MAX = 4.0         # only bonds shorter than this are drawn
NX, NY = 18, 18          # in-plane repetitions along a1 and a2

# ---------------------------------------------------------------------------


def read(path):
    """ids, positions and cell of the atoms of ATOM_TYPE."""
    lines = open(path).read().splitlines()
    box, tilt, start = {}, [0.0, 0.0, 0.0], None
    for n, line in enumerate(lines):
        words = line.split("#")[0].split()
        if not words:
            continue
        if words[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            box[words[-2][0]] = (float(words[0]), float(words[1]))
        elif words[-3:] == ["xy", "xz", "yz"]:
            tilt = [float(v) for v in words[:3]]
        elif words[0] == "Atoms":
            start = n + 1

    rows = []
    for line in lines[start:]:
        words = line.split("#")[0].split()
        if not words:
            if rows:
                break
            continue
        if not words[0].isdigit():
            break
        rows.append(words[:5])

    data = np.array(rows, dtype=float)
    keep = data[:, 1].astype(int) == ATOM_TYPE
    origin = np.array([box["x"][0], box["y"][0], box["z"][0]])
    cell = np.array([[box["x"][1] - box["x"][0], 0.0, 0.0],
                     [tilt[0], box["y"][1] - box["y"][0], 0.0],
                     [tilt[1], tilt[2], box["z"][1] - box["z"][0]]])
    print(f"{path}: {len(data)} atoms, {keep.sum()} of type {ATOM_TYPE}")
    return data[keep, 0].astype(int), data[keep, 2:5] - origin, cell


def tile(positions, cell, nx, ny, wrap=None):
    """Replicate in plane.  Returns positions, enlarged cell and the wrap used.

    Atoms are folded into the cell first, otherwise anything sitting outside
    the box would land on top of a copy of itself in the neighbouring tile.
    Pass the wrap returned by the unrelaxed call back in for the relaxed one,
    so both structures are folded by the same integer lattice shift.

    The ordering is tile-major: all atoms of tile 0, then all of tile 1, ...
    so two structures tiled the same way stay paired index by index.
    """
    frac = positions @ np.linalg.inv(cell)
    if wrap is None:
        wrap = np.floor(frac[:, :2])
    frac[:, :2] -= wrap
    wrapped = frac @ cell
    shifts = [m * cell[0] + n * cell[1] for m in range(nx) for n in range(ny)]
    big = np.concatenate([wrapped + shift for shift in shifts])
    return big, np.array([nx * cell[0], ny * cell[1], cell[2]]), wrap


def bond_vectors(positions, cell, pairs):
    """Bond vectors with the minimum image convention (wrapping-proof)."""
    frac = positions @ np.linalg.inv(cell)
    i, j = pairs.T
    delta = frac[j] - frac[i]
    delta[:, :2] -= np.round(delta[:, :2])
    return (delta @ cell)[:, :2]


if __name__ == "__main__":

    # Read the unrelaxed and relaxed structures
    ids_u, pos_u, cell_u = read(UNRELAXED)
    ids_r, pos_r, cell_r = read(RELAXED)
    cell0 = cell_r.copy()

    # Order the ids
    if not np.array_equal(ids_u, ids_r):
        order = np.argsort(ids_r)[np.searchsorted(ids_r[np.argsort(ids_r)], ids_u)]
        pos_r = pos_r[order]

    # an atom that crossed the boundary during relaxation comes back on the
    # far side of the box; fold it onto the image nearest its unrelaxed self
    delta = (pos_r - pos_u) @ np.linalg.inv(cell_r)
    jumped = np.abs(delta[:, :2]) > 0.5
    delta[:, :2] -= np.round(delta[:, :2])
    pos_r = pos_u + delta @ cell_r
    print(f"folded {jumped.any(axis=1).sum()} atoms back across the boundary")

    # replicate in plane, after the ids have been matched
    n_cell = len(pos_u)
    pos_u, cell_u, wrap = tile(pos_u, cell_u, NX, NY)
    pos_r, cell_r, _ = tile(pos_r, cell_r, NX, NY, wrap)
    print(f"tiled {NX} x {NY}: {n_cell} -> {len(pos_u)} atoms")

    # relaxation should move atoms by a fraction of an Angstrom
    moved = np.linalg.norm((pos_r - pos_u)[:, :2], axis=1)
    print(f"largest displacement {moved.max():.3f} A")

    # neighbours in the unrelaxed structure, periodic images included
    frac = pos_u @ np.linalg.inv(cell_u)
    pairs = []
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            offset = np.array([n1, n2, 0.0])
            vectors = ((frac[None, :, :] + offset) - frac[:, None, :]) @ cell_u
            distance = np.linalg.norm(vectors[:, :, :2], axis=2)
            for i, j in zip(*np.where((distance < CUTOFF) & (distance > 1e-6))):
                pairs.append((i, j))
    pairs = np.array(pairs)

    length_u = np.linalg.norm(bond_vectors(pos_u, cell_u, pairs), axis=1)
    length_r = np.linalg.norm(bond_vectors(pos_r, cell_r, pairs), axis=1)
    bond_strain = 100 * (length_r - length_u) / length_u

    strain = np.array([bond_strain[pairs[:, 0] == atom].mean() for atom in range(len(pos_u))])
    print(f"Min and max strain: {strain.min():.3f} and {strain.max():.3f} %")
    print(f"length_u: {length_u.min():.3f} and {length_u.max():.3f} A")
    print(f"length_r: {length_r.min():.3f} and {length_r.max():.3f} A")
    vmin = -1.0
    vmax = 0.7

    # every tile is a periodic copy, so all tiles must carry the same strain
    spread = np.abs(strain.reshape(NX * NY, n_cell) - strain[:n_cell]).max()
    print(f"tile-to-tile difference {spread:.2e} % (should be ~1e-12)")

    # --- plot --------------------------------------------------------------
    points = pos_r[:, :2]
    # draw only short bonds: anything wrapping round the outer cell is left out
    span = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
    inside = (span < DRAW_MAX) & (pairs[:, 0] < pairs[:, 1])
    segments = np.stack([points[pairs[inside, 0]], points[pairs[inside, 1]]], axis=1)
    print(f"drawing {inside.sum()} bonds shorter than {DRAW_MAX} A")

    fig, axis = plt.subplots(figsize=(3, 2.5))
    axis.add_collection(LineCollection(segments, colors="0.85", linewidths=1.25, zorder=0))
    dots = axis.scatter(points[:, 0], points[:, 1], c=strain, cmap="RdBu_r",
                        vmin=vmin, vmax=vmax, s=5, zorder=2)

    # Pristine unit cell
    a1, a2 = cell0[0, :2], cell0[1, :2]
    corners = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
    middle = corners + (NX // 2) * a1 + (NY // 2) * a2
    axis.plot(middle[:, 0], middle[:, 1], color="k", lw=1.75, alpha=0.5, zorder=3)

    xmin = 110; ymin = 40
    axis.set_xlim(xmin, xmin+110)
    axis.set_ylim(ymin, ymin+90)
    # fig.colorbar(dots, ax=axis, label="local strain (%)", fraction=0.046)
    axis.set_aspect("equal")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(f"strain_map_{ATOM_TYPE}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote strain_map_{ATOM_TYPE}.png")
    plt.show()
