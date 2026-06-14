"""
Quadrupolar Exciton Phase Diagram — three-phase competition with moiré penalty.
See our main paper for details and discussion.

Three competing phases:
  Q△  Quadrupolar triangular  — symmetric excitons on triangular lattice
  D□  Dipolar square          — staggered dipolar on square lattice
  D△  Dipolar triangular      — staggered dipolar on triangular lattice

Moiré potential V_M [meV] penalises D□ (non-triangular) by adding V_M per
site to its total energy. D△ is triangular and commensurate with the moiré,
so it pays no penalty.

Dimensionless axes:
  x : nd²   (density × interlayer separation²)
  y : R  =  ΔDQ / (e²/κd)

Ground state at each (nd², R):
  argmin { Ẽ_Q△,  Ẽ_D□ + R + Ṽ_M,  Ẽ_D△ + R }
  where Ṽ_M = V_M / (e²/κd)  and  Ẽ = E / (e²/κd).
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

E2_NM = 1.44   # e²/4πε₀  [eV·nm]


# PARAMETERS
@dataclass
class Params:
    """
    d            [nm]   interlayer separation 
    kappa        [-]    effective dielectric constant
                        Using kappa = 4 (hBN encapsulation); 
                         
    Delta_DQ     [meV]  single-particle quadrupole-dipole gap
                        Roughly tunneling energy scale; (E_q^+ - E_q^-)/2.
    Delta_DQ_max [meV]  upper ΔDQ limit → sets R_max on y-axis
    V_moire      [meV]  moiré penalty per site for non-triangular phases 
                        If triangular phases, they pay no penalty. 
                        This is a constant energy to make the distinction to clear.
    a_moire      [nm]   moiré lattice constant (None = free excitons only)
                        Not used at the moment, but could be used to mark moiré filling fractions on the x-axis.
    """
    d:            float      = 0.61
    kappa:        float      = 4.0
    Delta_DQ:     float      = 12.0
    Delta_DQ_max: float      = 20.0
    V_moire:      float      = 0.0    # meV — moiré penalty for D□
    a_moire:      float|None = None

    @property
    def e2_kappa(self):
        return E2_NM / self.kappa           # [eV·nm]

    @property
    def energy_scale(self):
        return self.e2_kappa / self.d       # e²/κd  [eV]

    @property
    def energy_scale_meV(self):
        return self.energy_scale * 1e3      # [meV]

    @property
    def R(self):
        """
        Dimensionless y-axis  R = ΔDQ / (e²/κd).
        """
        return (self.Delta_DQ * 1e-3) / self.energy_scale

    @property
    def V_moire_dim(self):
        """
        Dimensionless moiré penalty  Ṽ_M = V_M / (e²/κd).
        """
        return (self.V_moire * 1e-3) / self.energy_scale


@dataclass
class SystemPoint:
    """A specific (material, density) point overlaid on the phase diagram."""
    params: Params
    n:      float   # exciton density [nm⁻²]
    label:  str
    color:  str = 'red'
    marker: str = '*'

    @property
    def nd2(self):
        return self.n * self.params.d**2

    @property
    def R(self):
        return self.params.R


# PAIR POTENTIALS  (Main paper)
# Eqn. 5 of the main paper: Classical electrostatics.
def vpp(r, p):
    """Parallel dipole-dipole potential [eV]. Always repulsive."""
    return p.e2_kappa * (2.0/r - 2.0/np.sqrt(r**2 + p.d**2))

def vpm(r, p):
    """Anti-parallel dipole-dipole potential [eV]. Attractive at intermediate r."""
    return p.e2_kappa * (1.0/r + 1.0/np.sqrt(r**2 + (2*p.d)**2)
                         - 2.0/np.sqrt(r**2 + p.d**2))

def vsym(r, p):
    """Quadrupolar potential = ½(v++ + v+−) [eV]. Decays as r⁻⁵.
       Lower branch"""
    return 0.5 * (vpp(r, p) + vpm(r, p))


# LATTICE GEOMETRY  (precomputed dimensionless; scaled at runtime)
_N  = 65
_ns = np.arange(-_N, _N + 1)
_n1, _n2 = (x.ravel() for x in np.meshgrid(_ns, _ns))
_mask = ~((_n1 == 0) & (_n2 == 0))          # exclude origin
_n1, _n2 = _n1[_mask], _n2[_mask]

_a1 = np.array([1.0, 0.0])
_a2 = np.array([0.5, np.sqrt(3)/2])

# Triangular lattice: |n1·a1 + n2·a2|  (unit cell = 1)
_tri_d0 = np.linalg.norm(_n1[:, None]*_a1 + _n2[:, None]*_a2, axis=1)
# Square lattice: |n1·x + n2·y|
_sq_d0  = np.hypot(_n1.astype(float), _n2.astype(float))

# Sublattice parity for staggered ordering
#   D□: (n1+n2) even → v++,  odd → v+−
#   D△: n2 even      → v++,  odd → v+−  (alternating rows on triangular lattice)
_sq_sub  = (_n1 + _n2) % 2
_tri_sub = _n2 % 2

# 
# MADELUNG SUMS — interaction energy per site  [eV]
def _a_tri(n): return np.sqrt(2.0 / (np.sqrt(3) * n))   # triangular lattice constant
def _b_sq(n):  return 1.0 / np.sqrt(n)                   # square lattice constant

def E_sym_tri(n, p):
    """Q△: quadrupolar (symmetric) on triangular lattice."""
    return np.sum(vsym(_tri_d0 * _a_tri(n), p))

def E_stg_sq(n, p):
    """D□: staggered dipolar on square lattice."""
    r = _sq_d0 * _b_sq(n)
    return np.sum(np.where(_sq_sub == 0, vpp(r, p), vpm(r, p)))

def E_stg_tri(n, p):
    """D△: staggered dipolar on triangular lattice (row-alternating pattern)."""
    r = _tri_d0 * _a_tri(n)
    return np.sum(np.where(_tri_sub == 0, vpp(r, p), vpm(r, p)))


# ─────────────────────────────────────────────────────────────────
# THREE-PHASE COMPETITION
# ─────────────────────────────────────────────────────────────────
def build_phase_diagram(p, nd2v, Rax):
    """
    Compute the ground-state phase at every (nd², R) grid point.

    Returns
    -------
    Ph       : int array (len(Rax), len(nd2v))
                 0 → Q△,  1 → D□,  2 → D△
    Eq       : Madelung energy array for Q△  [eV]
    Ed_sq    : Madelung energy array for D□  [eV]
    Ed_tri   : Madelung energy array for D△  [eV]
    nd2d_sq  : nd² at minimum of Ed_sq (D□ droplet density)
    nd2d_tri : nd² at minimum of Ed_tri (D△ droplet density)

    Phase rule
    ----------
    Total effective energy per site:
      Q△  : E_Q(n)
      D□  : E_{D□}(n)  + ΔDQ  + V_M   [pays DQ gap + moiré penalty]
      D△  : E_{D△}(n)  + ΔDQ           [pays DQ gap only]

    Droplet construction for each dipolar branch:
      Below n*_□ (resp. n*_△), excitons condense into dense droplets;
      the effective interaction energy is clamped to its minimum value
      E_min (density-independent floor).

    In dimensionless units:
      GS = argmin { Ẽ_Q,  Ẽ_{D□} + R + Ṽ_M,  Ẽ_{D△} + R }
    """
    nv = nd2v / p.d**2

    print("  Computing Madelung sums (electrostatic costs) …")
    Eq     = np.array([E_sym_tri(n, p) for n in nv])
    Ed_sq  = np.array([E_stg_sq (n, p) for n in nv])
    Ed_tri = np.array([E_stg_tri(n, p) for n in nv])
    print("  Done.")

    # Droplet construction: floor each dipolar branch at its energy minimum
    idx_sq  = np.argmin(Ed_sq);  Emin_sq  = Ed_sq[idx_sq];  nd2d_sq  = nd2v[idx_sq]
    idx_tri = np.argmin(Ed_tri); Emin_tri = Ed_tri[idx_tri]; nd2d_tri = nd2v[idx_tri]

    print(f"  D□  minimum: nd²={nd2d_sq:.4f}, E_min={Emin_sq*1e3:.2f} meV")
    print(f"  D△  minimum: nd²={nd2d_tri:.4f}, E_min={Emin_tri*1e3:.2f} meV")

    ND, RR = np.meshgrid(nd2v, Rax)

    # Q△ — no gap cost, no moiré penalty, no droplet construction needed
    # Quadrupolar excitons on a triangular lattice; 
    Eq_g = np.interp(ND, nd2v, Eq)

    # D□ — gap R + moiré penalty Ṽ_M; droplet floor below nd2d_sq
    # Moiré penalty for being square; Also, converting quadrupole to dipole consts energy
    Ed_sq_g  = (np.where(ND <= nd2d_sq,  Emin_sq,
                         np.interp(ND, nd2v, Ed_sq))
                + RR + p.V_moire_dim)

    # D△ — gap R only; droplet floor below nd2d_tri
    Ed_tri_g = (np.where(ND <= nd2d_tri, Emin_tri,
                         np.interp(ND, nd2v, Ed_tri))
                + RR)

    # Ground state: 3-way argmin  (0=Q△, 1=D□, 2=D△)
    Ph3 = np.argmin(np.stack([Eq_g, Ed_sq_g, Ed_tri_g], axis=0), axis=0)

    # Expand each dipolar branch into droplet vs uniform sub-regions.
    #   0 → Q△            (unchanged)
    #   1 and nd² ≤ nd2d_sq → D□ droplet  (1)
    #   1 and nd² >  nd2d_sq → D□ uniform  (2)
    #   2 and nd² ≤ nd2d_tri → D△ droplet  (3)
    #   2 and nd² >  nd2d_tri → D△ uniform  (4)
    Ph = np.where(Ph3 == 0, 0,
         np.where(Ph3 == 1,
             np.where(ND <= nd2d_sq,  1, 2),
             np.where(ND <= nd2d_tri, 3, 4)))

    return Ph, Eq, Ed_sq, Ed_tri, nd2d_sq, nd2d_tri


# ─────────────────────────────────────────────────────────────────
# MOIRÉ FILLING FRACTIONS
# ─────────────────────────────────────────────────────────────────
def moire_fillings(p, m_max=1):
    """nd² values for 1 exciton per m×m moiré supercell."""
    ms    = np.arange(1, m_max + 1)
    a_exc = ms * p.a_moire
    n_exc = 2.0 / (np.sqrt(3) * a_exc**2)
    return ms, n_exc * p.d**2


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def run(p=None, points=None):

    # Default parameters
    if p is None:
        p = Params()
    if points is None:
        points = []

    # Print parameters
    print(f"\n{'─'*60}")
    print(f"  d = {p.d} nm   κ = {p.kappa}")
    print(f"  e²/κd  = {p.energy_scale_meV:.1f} meV")
    print(f"  ΔDQ    = {p.Delta_DQ} meV   →   R = {p.R:.5f}")
    print(f"  V_M    = {p.V_moire} meV   →   Ṽ_M = {p.V_moire_dim:.5f}")
    if p.a_moire:
        print(f"  a_moiré = {p.a_moire} nm")
    print(f"{'─'*60}")

    # Density grid
    a_min_nm = p.a_moire if p.a_moire else 1.5
    n_max    = 2.0 / (np.sqrt(3) * a_min_nm**2)
    nd2_max  = n_max * p.d**2
    nd2v     = np.logspace(-2.5, np.log10(nd2_max), 500)

    # R axis (R is sensitive to twist angle dependence as well as hBN screening)
    R_max = (p.Delta_DQ_max * 1e-3) / p.energy_scale
    R_min = (1.0           * 1e-3) / p.energy_scale
    Rax   = np.linspace(R_min, R_max, 400)

    Ph, Eq, Ed_sq, Ed_tri, nd2d_sq, nd2d_tri = build_phase_diagram(p, nd2v, Rax)

    for i, name in enumerate(['Q△', 'D□ droplet', 'D□ uniform', 'D△ droplet', 'D△ uniform']):
        frac = np.mean(Ph == i) * 100
        if frac > 0:
            print(f"  {name}: {frac:.1f}% of diagram area")

    # ── Colours ──────────────────────────────────────────────────
    C_Q_TRI    = '#648FFF'   # blue       — Q△
    C_D_DROP   = '#FFB000'   # amber      — D□ droplet
    C_D_UNIF   = '#FE6100'   # orange-red — D□ uniform staggered
    C_D_TRI_DR = '#6EE7CF'   # light teal — D△ droplet
    C_D_TRI_UN = '#0D9488'   # dark teal  — D△ uniform staggered
    cmap = mc.ListedColormap([C_Q_TRI, C_D_DROP, C_D_UNIF, C_D_TRI_DR, C_D_TRI_UN])

    # ── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.0, 3.0), facecolor='white')
    fig.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.13)

    ax.set_facecolor('#F8F9FA')
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color('#CCCCCC')

    ax.pcolormesh(nd2v, Rax, Ph, cmap=cmap, vmin=0, vmax=4,
                  alpha=0.85, shading='auto', rasterized=True)

    # Phase boundaries between all five regions
    ax.contour(nd2v, Rax, Ph, levels=[0.5, 1.5, 2.5, 3.5],
               colors=['#333333'], linewidths=[1.0], alpha=0.6)

    # Moiré filling markers
    if p.a_moire:
        ms, nd2_m = moire_fillings(p)
        for m, x in zip(ms, nd2_m):
            if nd2v[0] < x <= nd2_max:
                ax.axvline(x, color='white', lw=0.8, ls=':', alpha=0.6, zorder=5)
                ax.text(x, Rax[-1]*0.97, f'm={m}', fontsize=6, color='white',
                        ha='center', va='top', zorder=6)

    # Overlay system points
    for sp in points:
        ax.plot(sp.nd2, sp.R, sp.marker, color=sp.color,
                ms=12, markeredgecolor='k', markeredgewidth=0.8,
                zorder=10, label=sp.label)

    fs = 11
    ax.set_xlabel(r'$nd^2$', fontsize=fs)
    #ax.set_ylabel(r'$R = \dfrac{(\mathcal{E}^{+}_{q} - \mathcal{E}^{-}_{q})\, 4\pi\epsilon_0 \epsilon_{\mathrm{eff}} d}{2e^2}$', fontsize=fs)
    ax.set_xlim(nd2v[0], nd2_max)
    ax.set_ylim(Rax[0], Rax[-1])
    ax.set_yticks([0.01, 0.02, 0.03])
    #ax.set_yticklabels(['0.01', '0.02', '0.03'], fontsize=fs)
    ax.set_yticklabels([])
    ax.set_xticks([0.05, 0.1, 0.15])
    ax.set_xticklabels(['0.05', '0.1', '0.15'], fontsize=fs)

    legend_patches = [
        Patch(color=C_Q_TRI,    alpha=0.85),
        Patch(color=C_D_DROP,   alpha=0.85),
        Patch(color=C_D_UNIF,   alpha=0.85),
        Patch(color=C_D_TRI_DR, alpha=0.85),
        Patch(color=C_D_TRI_UN, alpha=0.85),
    ]
    pt_handles = [
        Line2D([], [], color=sp.color, marker=sp.marker, ms=8, lw=0,
               markeredgecolor='k', label=sp.label)
        for sp in points
    ]
    # ax.legend(handles=legend_patches + pt_handles,
    #           fontsize=7.5, loc='upper right', framealpha=0.88)

    V_str = f'$V_M$={p.V_moire} meV   $\\tilde{{V}}_M$={p.V_moire_dim:.4f}'

    fname = ('quadrupolar_phase_moire_confined.png'
             if p.a_moire else 'quadrupolar_phase_diagram.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight',
                transparent=True, pad_inches=0.1)
    plt.show()
    print(f"\n  Figure saved → {fname}")

    print_math(p)
    return Ph, Rax, nd2v


# ─────────────────────────────────────────────────────────────────
# MATH SUMMARY
# ─────────────────────────────────────────────────────────────────
def print_math(p):
    es = p.energy_scale_meV
    print(f"""
  Current parameters:
    e²/κd  = {es:.1f} meV
    V_M    = {p.V_moire:.2f} meV   →   Ṽ_M = {p.V_moire_dim:.5f}
    ΔDQ    = {p.Delta_DQ:.1f} meV  →   R   = {p.R:.5f}
""")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    paper = Params(
        d            = 0.61,   # nm
        kappa        = 4.0,
        Delta_DQ     = 12.0,   # meV
        Delta_DQ_max = 20.0,   # meV (sets R_max)
        V_moire      = 25.0,    # meV — penalises D□; try 0→5→8→12→15
    )
    run(paper)
