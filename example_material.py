"""example_material.py — Showcase of the new Material class.

Demonstrates all major usage patterns:
  1.  Empty init + set_lattice + compute_reflections (q-range)
  2.  from_lattice_parameters with wavelength + 2θ range
  3.  from_lattice_parameters with explicit hkl list
  4.  from_cif with energy + 2θ range
  5.  from_cif with q range only (no wavelength needed for listing d-spacings)
  6.  Filtering, inspecting reflections
  7.  Operator-facing attributes (h_vecs, point_group_matrices)
  8.  Incremental load_cif example

Run from the workspace root:
    python example_material.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from diffractom.crystallography.material import Material

SEP = "─" * 60


def banner(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print("═" * 60)


def show_reflections(mat: Material, n: int = 8) -> None:
    """Print first n reflections in a table."""
    print(f"  {'HKL':>12}  {'d (Å)':>8}  {'2θ (°)':>8}  {'I (norm)':>9}  {'mult':>5}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*5}")
    for i in range(min(n, len(mat))):
        r = mat.reflection(i)
        hkl_str = str(r['hkl'])
        tth_deg = np.degrees(r['two_theta'])
        print(f"  {hkl_str:>12}  {r['d_spacing']:8.4f}  "
              f"{tth_deg:8.3f}  {r['intensity']:9.2f}  {r['multiplicity']:5d}")
    if len(mat) > n:
        print(f"  ... ({len(mat) - n} more)")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty init → set_lattice → compute_reflections (q range)
# ─────────────────────────────────────────────────────────────────────────────
banner("1 · Empty init + set_lattice + compute_reflections (q range)")

mat = Material(name="Iron (BCC)")

# BCC iron: a = 2.866 Å, cubic
mat.set_lattice(a=2.866, b=2.866, c=2.866)
print(f"  crystal system inferred : {mat.crystal_system}")
print(f"  lattice params          : {mat.lattice_params}")

# Compute reflections for q = 1 … 10 Å⁻¹ (physics, 2π/d)
# Wavelength optional here — 2θ will be stored as 0 if not given
mat.compute_reflections(q_min=1.0, q_max=10.0)
print(f"  reflections found       : {len(mat)}")
show_reflections(mat)

# ─────────────────────────────────────────────────────────────────────────────
# 2. from_lattice_parameters — hexagonal phase (e.g. α-Fe₂O₃ hematite)
#    with wavelength + 2θ range
# ─────────────────────────────────────────────────────────────────────────────
banner("2 · from_lattice_parameters  (hexagonal, wavelength + 2θ range)")

mat2 = Material.from_lattice_parameters(
    a=5.035, c=13.747,
    gamma=120.0,           # explicit hexagonal γ angle
    symmetry_group='hexagonal',
    wavelength_A=0.2478,   # 50 keV
    min_two_theta=np.deg2rad(2.0),
    max_two_theta=np.deg2rad(25.0),
    name="Hematite lattice only",
)
print(f"  crystal system : {mat2.crystal_system}")
print(f"  reflections    : {len(mat2)}")
show_reflections(mat2)

# ─────────────────────────────────────────────────────────────────────────────
# 3. from_lattice_parameters — explicit hkl list
# ─────────────────────────────────────────────────────────────────────────────
banner("3 · from_lattice_parameters  (explicit hkl list, energy in keV)")

FCC_HKL = [(1,1,1),(2,0,0),(2,2,0),(3,1,1),(2,2,2),(4,0,0),(3,3,1),(4,2,0)]

mat3 = Material.from_lattice_parameters(
    a=3.615,                # FCC copper
    symmetry_group='cubic',
    energy_keV=60.0,
    hkl_list=FCC_HKL,
    name="FCC Copper (selected peaks)",
)
print(f"  crystal system : {mat3.crystal_system}")
print(f"  reflections    : {len(mat3)}")
show_reflections(mat3)

# ─────────────────────────────────────────────────────────────────────────────
# 4. from_cif — with energy + 2θ range
# ─────────────────────────────────────────────────────────────────────────────
banner("4 · from_cif  (energy + 2θ range)")

CIF_PATH = REPO / "input" / "test_cifs" / "82903_Fe2O3_hem.cif"
if CIF_PATH.exists():
    mat4 = Material.from_cif(
        CIF_PATH,
        energy_keV=50.0,
        min_two_theta=np.deg2rad(3.0),
        max_two_theta=np.deg2rad(30.0),
    )
    print(f"  name           : {mat4.name}")
    print(f"  space group    : {mat4.space_group_symbol} ({mat4.space_group_number})")
    print(f"  crystal system : {mat4.crystal_system}")
    print(f"  lattice params : a={mat4.lattice_params['a']:.4f}  "
          f"c={mat4.lattice_params['c']:.4f}")
    print(f"  reflections    : {len(mat4)}")
    show_reflections(mat4)
else:
    print(f"  [SKIP] CIF not found: {CIF_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. from_cif — q range only (d-spacings only, 2θ stored as 0)
# ─────────────────────────────────────────────────────────────────────────────
banner("5 · from_cif  (q range only, no wavelength needed)")

CIF_IRON = REPO / "input" / "test_cifs" / "27237_wustite.cif"
if CIF_IRON.exists():
    mat5 = Material.from_cif(
        CIF_IRON,
        q_min=1.0,
        q_max=12.0,
    )
    print(f"  name           : {mat5.name}")
    print(f"  space group    : {mat5.space_group_symbol} ({mat5.space_group_number})")
    print(f"  crystal system : {mat5.crystal_system}")
    print(f"  reflections    : {len(mat5)}")
    print(f"  {'HKL':>12}  {'d (Å)':>8}  {'I (norm)':>9}  {'mult':>5}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*5}")
    for i in range(min(8, len(mat5))):
        r = mat5.reflection(i)
        print(f"  {str(r['hkl']):>12}  {r['d_spacing']:8.4f}  "
              f"{r['intensity']:9.2f}  {r['multiplicity']:5d}")
else:
    print(f"  [SKIP] CIF not found: {CIF_IRON}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Filtering
# ─────────────────────────────────────────────────────────────────────────────
banner("6 · Filtering  (keep only strong peaks)")

if CIF_PATH.exists():
    mat6 = Material.from_cif(
        CIF_PATH,
        energy_keV=50.0,
        min_two_theta=np.deg2rad(3.0),
        max_two_theta=np.deg2rad(30.0),
    )
    print(f"  Before filter: {len(mat6)} reflections")
    mat6.filter_by_intensity(min_intensity=5.0)   # keep I ≥ 5% of max
    print(f"  After  filter (I ≥ 5): {len(mat6)} reflections")
    show_reflections(mat6)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Operator-facing attributes
# ─────────────────────────────────────────────────────────────────────────────
banner("7 · Operator-facing attributes")

if CIF_PATH.exists():
    print(f"\n  mat4.hkl.shape              = {mat4.hkl.shape}")
    print(f"  mat4.h_vecs.shape           = {mat4.h_vecs.shape}")
    print(f"  mat4.h_vecs_normed.shape    = {mat4.h_vecs_normed.shape}")
    print(f"  mat4.point_group_matrices.shape = {mat4.point_group_matrices.shape}")
    print(f"  mat4.num_sym_ops            = {mat4.num_sym_ops}")
    print(f"\n  First 3 h-vectors (Å⁻¹, with 2π):")
    for i in range(min(3, len(mat4))):
        hkl = tuple(mat4.hkl[i])
        hv  = mat4.h_vecs[i]
        print(f"    {str(hkl):>14} → [{hv[0]:+.4f}, {hv[1]:+.4f}, {hv[2]:+.4f}]  "
              f"  |h| = {np.linalg.norm(hv):.4f} Å⁻¹")
    print(f"\n  Point group rotation matrices (shape = {mat4.point_group_matrices.shape}):")
    print(f"  First matrix (flattened  = 9 floats):")
    print("   ", mat4.point_group_matrices[0])

# ─────────────────────────────────────────────────────────────────────────────
# 8. Incremental construction with load_cif
# ─────────────────────────────────────────────────────────────────────────────
banner("8 · Incremental construction  (load_cif + compute_reflections separately)")

if CIF_PATH.exists():
    mat8 = Material()
    mat8.load_cif(CIF_PATH)
    print(f"  After load_cif:")
    print(f"    name    = {mat8.name}")
    print(f"    system  = {mat8.crystal_system}")
    print(f"    SG      = {mat8.space_group_symbol} ({mat8.space_group_number})")
    print(f"    basis sites = {len(mat8._basis_frac) if mat8._basis_frac is not None else 0}")
    print(f"    reflections = {len(mat8)}  (not yet computed)")

    mat8.compute_reflections(
        q_min=1.5, q_max=8.0,
        wavelength_A=0.2478,
    )
    print(f"  After compute_reflections (q=1.5–8 Å⁻¹):")
    print(f"    reflections = {len(mat8)}")
    show_reflections(mat8, n=5)

# ─────────────────────────────────────────────────────────────────────────────
# 9. summary()
# ─────────────────────────────────────────────────────────────────────────────
banner("9 · summary()")

for m, label in [(mat, "Iron BCC (lattice only)"),
                 (mat3, "FCC Copper (explicit HKLs)")]:
    print(f"\n  {label}:")
    for k, v in m.summary().items():
        print(f"    {k:20s} = {v}")

print(f"\n{'═'*60}")
print("  All examples completed successfully.")
print("═" * 60)
