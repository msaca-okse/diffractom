"""test_material_vs_pymatgen.py — Compare new Material class against old pymatgen-based one.

Run from the workspace root:
    python test_material_vs_pymatgen.py

Requires:
    conda2 / conda activate textom  (for access to pymatgen)
    The new material class must be installed (pip install -e . from repo root).
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
CIF_DIR = REPO / "input" / "test_cifs"

sys.path.insert(0, str(REPO / "src"))

# ── new Material ────────────────────────────────────────────────────────────
from diffractom.crystallography.material import Material as NewMaterial
from diffractom.crystallography.cif_parser import parse_cif

# ── old Material (kept as material_old.py) ──────────────────────────────────
from diffractom.crystallography.material_old import Material as OldMaterial

# ── test settings ────────────────────────────────────────────────────────────
ENERGY_KEV     = 50.0            # 50 keV synchrotron beam
WAVELENGTH_A   = 12.39842 / ENERGY_KEV
MIN_TTH        = np.deg2rad(2.0)
MAX_TTH        = np.deg2rad(30.0)
D_TOL          = 1e-3            # relative tolerance for d-spacing comparison


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_hkl(hkl):
    """Canonical form: ensure first non-zero index is positive."""
    h = list(hkl)
    for x in h:
        if x != 0:
            if x < 0:
                h = [-v for v in h]
            break
    return tuple(h)


def _hkl_set(mat):
    """Return set of canonical HKL tuples from a material."""
    return {_normalise_hkl(tuple(int(x) for x in row)) for row in mat.hkls()}


def _d_dict(mat):
    """Return dict {canonical_hkl: d_spacing}."""
    d = {}
    for row, d_val in zip(mat.hkls(), mat.d_spacings()):
        key = _normalise_hkl(tuple(int(x) for x in row))
        d[key] = float(d_val)
    return d


def _mult_dict(mat):
    """Return dict {canonical_hkl: multiplicity}."""
    m = {}
    for row, mult in zip(mat.hkls(), mat.multiplicities()):
        key = _normalise_hkl(tuple(int(x) for x in row))
        m[key] = int(mult)
    return m


def _compare(cif_path: Path, verbose: bool = True) -> dict:
    """Run comparison for one CIF file.  Returns a result dict."""
    result = {
        "file":           cif_path.name,
        "old_ok":         False,
        "new_ok":         False,
        "lattice_match":  None,
        "sg_match":       None,
        "hkl_overlap":    None,     # fraction of old HKLs found in new
        "d_match":        None,     # fraction with matching d-spacings
        "mult_match":     None,     # fraction with matching multiplicities
        "errors":         [],
    }

    # ── Parse CIF (new parser) for ground-truth lattice params ──────────────
    try:
        cif_data = parse_cif(cif_path)
    except Exception as e:
        result["errors"].append(f"CIF parse error: {e}")
        return result

    lp = {k: cif_data[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")}

    # ── Old Material ─────────────────────────────────────────────────────────
    old_mat = None
    try:
        old_mat = OldMaterial.from_cif(
            str(cif_path),
            wavelength_kev=ENERGY_KEV,
            min_two_theta=MIN_TTH,
            max_two_theta=MAX_TTH,
        )
        result["old_ok"] = True
    except Exception as e:
        result["errors"].append(f"Old Material failed: {type(e).__name__}: {e}")

    # ── New Material ──────────────────────────────────────────────────────────
    new_mat = None
    try:
        new_mat = NewMaterial.from_cif(
            cif_path,
            wavelength_kev=ENERGY_KEV,
            min_two_theta=MIN_TTH,
            max_two_theta=MAX_TTH,
        )
        result["new_ok"] = True
    except Exception as e:
        result["errors"].append(f"New Material failed: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()

    if not result["new_ok"]:
        return result

    # ── Lattice parameter comparison (new vs CIF ground truth) ───────────────
    new_lp = new_mat.lattice_params
    lp_diffs = {k: abs(new_lp[k] - lp[k]) for k in lp}
    lp_ok = all(v < 0.01 for v in lp_diffs.values())
    result["lattice_match"] = lp_ok
    if verbose and not lp_ok:
        print(f"  [WARN] lattice mismatch: {lp_diffs}")

    # ── Space group (new vs CIF) ──────────────────────────────────────────────
    sg_cif = cif_data.get("space_group_number")
    sg_new = new_mat.space_group_number
    result["sg_match"] = (sg_cif == sg_new) if sg_cif is not None else None
    if verbose and sg_cif is not None and not result["sg_match"]:
        print(f"  [WARN] SG: CIF={sg_cif}, new={sg_new}")

    # ── HKL / d-spacing / multiplicity comparison (old vs new) ───────────────
    if not result["old_ok"]:
        if verbose:
            print(f"  [SKIP] old Material unavailable — skipping HKL comparison")
        return result

    old_hkls = _hkl_set(old_mat)
    new_hkls = _hkl_set(new_mat)
    common   = old_hkls & new_hkls

    n_old = len(old_hkls)
    n_new = len(new_hkls)
    n_common = len(common)
    overlap = n_common / n_old if n_old > 0 else 0.0
    result["hkl_overlap"] = overlap

    if verbose:
        print(f"  HKLs: old={n_old}, new={n_new}, common={n_common}  "
              f"(overlap = {overlap:.1%})")

    # ── d-spacing comparison ──────────────────────────────────────────────────
    old_d = _d_dict(old_mat)
    new_d = _d_dict(new_mat)
    d_ok = 0
    d_total = 0
    for hkl in common:
        if hkl in old_d and hkl in new_d:
            rel_err = abs(old_d[hkl] - new_d[hkl]) / (old_d[hkl] + 1e-12)
            if rel_err < D_TOL:
                d_ok += 1
            elif verbose:
                print(f"  [WARN] d-spacing mismatch {hkl}: "
                      f"old={old_d[hkl]:.5f} new={new_d[hkl]:.5f} rel={rel_err:.2e}")
            d_total += 1
    result["d_match"] = d_ok / d_total if d_total > 0 else None
    if verbose:
        print(f"  d-spacing match: {d_ok}/{d_total}")

    # ── multiplicity comparison ───────────────────────────────────────────────
    old_mult = _mult_dict(old_mat)
    new_mult = _mult_dict(new_mat)
    mult_ok = 0
    mult_total = 0
    for hkl in common:
        if hkl in old_mult and hkl in new_mult:
            if old_mult[hkl] == new_mult[hkl]:
                mult_ok += 1
            elif verbose:
                print(f"  [WARN] mult mismatch {hkl}: "
                      f"old={old_mult[hkl]} new={new_mult[hkl]}")
            mult_total += 1
    result["mult_match"] = mult_ok / mult_total if mult_total > 0 else None
    if verbose:
        print(f"  multiplicity match: {mult_ok}/{mult_total}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cif_files = sorted(CIF_DIR.glob("*.cif"))
    if not cif_files:
        print(f"No CIF files found in {CIF_DIR}")
        return

    print(f"Testing {len(cif_files)} CIF files")
    print(f"  Energy: {ENERGY_KEV} keV  (λ = {WAVELENGTH_A:.5f} Å)")
    print(f"  2θ range: {np.degrees(MIN_TTH):.1f}° – {np.degrees(MAX_TTH):.1f}°")
    print("=" * 70)

    all_results = []
    for cif in cif_files:
        print(f"\n{cif.name}")
        print("-" * 60)
        res = _compare(cif, verbose=True)
        all_results.append(res)
        status = []
        if res["new_ok"]:
            status.append("NEW OK")
        else:
            status.append("NEW FAIL")
        if res["old_ok"]:
            status.append("OLD OK")
        else:
            status.append("OLD SKIP")
        print(f"  → {' | '.join(status)}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_new_ok    = sum(r["new_ok"] for r in all_results)
    n_both_ok   = sum(r["new_ok"] and r["old_ok"] for r in all_results)
    ov_vals     = [r["hkl_overlap"] for r in all_results if r["hkl_overlap"] is not None]
    d_vals      = [r["d_match"]     for r in all_results if r["d_match"]     is not None]
    mult_vals   = [r["mult_match"]  for r in all_results if r["mult_match"]  is not None]

    print(f"New Material parsed successfully : {n_new_ok}/{len(all_results)}")
    print(f"Both old+new available for comparison: {n_both_ok}/{len(all_results)}")
    if ov_vals:
        print(f"Mean HKL overlap (old ∩ new / old): {np.mean(ov_vals):.1%}  "
              f"(min={min(ov_vals):.1%})")
    if d_vals:
        print(f"Mean d-spacing match fraction      : {np.mean(d_vals):.1%}")
    if mult_vals:
        print(f"Mean multiplicity match fraction   : {np.mean(mult_vals):.1%}")

    # ── Per-file table ───────────────────────────────────────────────────────
    print()
    print(f"{'CIF file':<40} {'NEW':>5} {'OLD':>5} {'HKLov':>7} {'d':>7} {'mult':>7}")
    print("-" * 75)
    for r in all_results:
        hkl  = f"{r['hkl_overlap']:.0%}" if r['hkl_overlap'] is not None else "-"
        d    = f"{r['d_match']:.0%}"     if r['d_match'] is not None else "-"
        mult = f"{r['mult_match']:.0%}"  if r['mult_match'] is not None else "-"

        row = (
            f"{r['file'][:39]:<40} "
            f"{'OK' if r['new_ok'] else 'FAIL':>5} "
            f"{'OK' if r['old_ok'] else 'skip':>5} "
            f"{hkl:>7} "
            f"{d:>7} "
            f"{mult:>7}"
        )
        print(row)


if __name__ == "__main__":
    main()
