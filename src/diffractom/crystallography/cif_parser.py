"""cif_parser.py — Pure-Python CIF file parser.

Parses a CIF file and returns a dict with crystal structure information:
lattice parameters, space group, symmetry operations, and a fully-expanded
asymmetric unit (all atomic sites in the unit cell, including partial occupancies).

Ported and extended from input/helmholtz3d-app.js (parseCIF and helpers).
"""

import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _strip_quotes(s: str) -> str:
    """Strip balanced single or double quotes from a string."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _parse_cif_float(s: str) -> float:
    """Parse a CIF numeric, stripping the (esd) suffix if present, e.g. '5.4123(4)' → 5.4123."""
    s = s.strip()
    if s in ('?', '.', ''):
        raise ValueError(f"CIF value is unknown/missing: {s!r}")
    s = re.sub(r'\(\d+\)$', '', s)
    return float(s)


def _tokenize(line: str) -> list[str]:
    """Split one CIF line into string tokens respecting single- and double-quoted values."""
    tokens: list[str] = []
    s = line.strip()
    while s:
        if s[0] in ('"', "'"):
            q = s[0]
            end = s.find(q, 1)
            if end > 0:
                tokens.append(s[1:end])
                s = s[end + 1:].lstrip()
            else:
                tokens.append(s[1:])
                s = ''
        else:
            m = re.match(r'\S+', s)
            if m:
                tokens.append(m.group())
                s = s[m.end():].lstrip()
            else:
                break
    return tokens


# ---------------------------------------------------------------------------
# Symmetry operation parser (ported from helmholtz3d-app.js)
# ---------------------------------------------------------------------------

def _parse_sym_op(op_str: str) -> list[list[float]]:
    """Parse "x,y,z"-style symmetry operation into a 3×4 matrix [cx, cy, cz, offset]."""
    parts = op_str.replace("'", "").split(',')
    matrix: list[list[float]] = []
    for part in parts:
        cx = cy = cz = offset = 0.0
        s = part.replace(' ', '')
        i = 0
        sign = 1
        while i < len(s):
            c = s[i]
            if c == '+':
                sign = 1;  i += 1;  continue
            if c == '-':
                sign = -1; i += 1;  continue
            if c == 'x':
                cx = float(sign); sign = 1; i += 1; continue
            if c == 'y':
                cy = float(sign); sign = 1; i += 1; continue
            if c == 'z':
                cz = float(sign); sign = 1; i += 1; continue
            # Numeric part
            num_str = ''
            while i < len(s) and s[i] in '0123456789./':
                num_str += s[i]; i += 1
            if num_str:
                if '/' in num_str:
                    num_s, den_s = num_str.split('/')
                    val = float(num_s) / float(den_s)
                else:
                    val = float(num_str)
                # Check if number precedes a variable (e.g. "2x")
                if i < len(s) and s[i] in 'xyz':
                    v = s[i]; i += 1
                    if v == 'x':   cx = sign * val
                    elif v == 'y': cy = sign * val
                    elif v == 'z': cz = sign * val
                    sign = 1
                else:
                    offset += sign * val
                    sign = 1
        matrix.append([cx, cy, cz, offset])
    return matrix  # 3 rows, each [cx, cy, cz, offset]


def _apply_sym_op(matrix: list[list[float]], x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply a 3×4 symmetry matrix to fractional coordinate (x, y, z)."""
    return (
        matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z + matrix[0][3],
        matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z + matrix[1][3],
        matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z + matrix[2][3],
    )


def _mod1(v: float) -> float:
    """Fold a fractional coordinate into [0, 1)."""
    return ((v % 1.0) + 1.0) % 1.0


def _close_modulo(a: float, b: float, tol: float = 0.01) -> bool:
    """True if a ≈ b modulo 1 within tol."""
    d = abs(a - b)
    return d < tol or d > 1.0 - tol


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

_ANISO_U = ('11', '22', '33', '12', '13', '23')


def _parse_aniso(loops) -> dict:
    """Anisotropic displacements {label: (U11, U22, U33, U12, U13, U23)} in Å² from an aniso loop
    (U_ij, or B_ij converted with B = 8π² U). Empty if the CIF has none."""
    for loop in loops:
        cols = loop['columns']
        lab_keys = ('_atom_site_aniso_label', '_atom_site_aniso.label')
        i_lab = next((cols.index(k) for k in lab_keys if k in cols), None)
        if i_lab is None:
            continue
        for kind, scale in (('U', 1.0), ('B', 1.0 / (8.0 * np.pi ** 2))):
            idx = []
            for ij in _ANISO_U:
                keys = (f'_atom_site_aniso_{kind}_{ij}', f'_atom_site_aniso.{kind}_{ij}')
                idx.append(next((cols.index(k) for k in keys if k in cols), None))
            if all(i is not None for i in idx):
                out = {}
                for row in loop['rows_tokens']:
                    try:
                        out[row[i_lab]] = tuple(_parse_cif_float(row[i]) * scale for i in idx)
                    except (ValueError, IndexError):
                        continue
                return out
    return {}


def _u_equivalent(u6, cell) -> float:
    """U_eq = (1/3) sum_ij U_ij a*_i a*_j (a_i . a_j) (Fischer & Tillmanns 1988) from the CIF U_ij."""
    a, b, c, al, be, ga = cell
    ca, cb, cg = np.cos(np.radians([al, be, ga]))
    G = np.array([[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]])
    astar = np.sqrt(np.diag(np.linalg.inv(G)))
    U11, U22, U33, U12, U13, U23 = u6
    U = np.array([[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]])
    return float(np.einsum('ij,i,j,ij->', U, astar, astar, G) / 3.0)


def _ops_from_space_group(symbol, number) -> list:
    """All symmetry operations (incl. centring) of a space group as 'x,y,z' strings, from gemmi."""
    try:
        import gemmi
    except ImportError:
        raise ValueError(
            "The CIF lists no symmetry operations, only the space group "
            f"({symbol or number}). Add the operations to the CIF, or install gemmi "
            "(pip install gemmi) to generate them.") from None
    sg = None
    if symbol:
        sg = gemmi.find_spacegroup_by_name(symbol)
    if sg is None and number:
        sg = gemmi.find_spacegroup_by_number(int(number))
    if sg is None:
        raise ValueError(f"Unknown space group {symbol or number}.")
    return [op.triplet() for op in sg.operations()]


def expand_asymmetric_unit(asym_atoms: list[dict], sym_ops: list[str]):
    """Apply the symmetry operations ('x,y,z' strings) to the asymmetric unit (dicts with sym, fx, fy, fz, occ,
    uiso, label) and keep one copy of every position in the cell (special positions deduplicated).
    Returns (basis, elements, occupancies, uiso, labels) as lists."""
    parsed_ops = [_parse_sym_op(op) for op in sym_ops]

    basis_list:    list[list[float]] = []
    elements_list: list[str]         = []
    occ_list:      list[float]       = []
    uiso_list:     list[float]       = []
    label_list:    list[str]         = []

    for atom in asym_atoms:
        start = len(basis_list)
        for op in parsed_ops:
            pos = _apply_sym_op(op, atom['fx'], atom['fy'], atom['fz'])
            mx = _mod1(pos[0])
            my = _mod1(pos[1])
            mz = _mod1(pos[2])

            # duplicates only among the images of the same site (distinct sites may share a position: mixed
            # occupancy)
            is_dup = any(
                _close_modulo(b[0], mx) and _close_modulo(b[1], my) and _close_modulo(b[2], mz)
                for b in basis_list[start:]
            )
            if not is_dup:
                basis_list.append([mx, my, mz])
                elements_list.append(atom['sym'])
                occ_list.append(atom['occ'])
                uiso_list.append(atom.get('uiso', np.nan))
                label_list.append(atom.get('label', atom['sym']))
    return basis_list, elements_list, occ_list, uiso_list, label_list


def parse_cif(path) -> dict:
    """Parse a CIF file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict with keys:
        name            : str
        a, b, c         : float  (Å)
        alpha,beta,gamma: float  (degrees)
        space_group_number : int or None
        space_group_symbol : str or None
        sym_ops         : list[str]   — raw symmetry-operation strings
        basis           : np.ndarray  shape (N, 3)  fractional coordinates of full basis
        elements        : list[str]   — element symbol per site
        occupancies     : np.ndarray  shape (N,)
        uiso            : np.ndarray  shape (N,) — isotropic displacement U (Å²) per site: U_iso, or B_iso / 8π²,
                          or U_eq from anisotropic U_ij / B_ij; NaN where the CIF gives none
        labels          : list[str]   — atom-site label per site

    Symmetry: the operations listed in the CIF are used. A CIF without them but with a space-group symbol or
    number is expanded with the operations from gemmi (optional dependency); without gemmi this raises.
    """
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    kv: dict[str, str] = {}
    loops: list[dict] = []   # each: {columns: [...], rows_tokens: [[...], ...]}

    i = 0
    first_data_block_done = False

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Skip blank and comment lines
        if not line or line.startswith('#'):
            i += 1
            continue

        # data_ block marker
        if line.lower().startswith('data_'):
            if first_data_block_done:
                break   # stop at start of second data block
            first_data_block_done = True
            i += 1
            continue

        # ---- loop_ block ----
        if line == 'loop_':
            i += 1
            columns: list[str] = []
            # Collect column names (lines starting with _)
            while i < len(lines):
                col_line = lines[i].strip()
                if not col_line or col_line.startswith('#'):
                    i += 1
                    continue
                if col_line.startswith('_'):
                    # Take only the first word (column key)
                    columns.append(col_line.split()[0])
                    i += 1
                else:
                    break

            # Collect data tokens (until next CIF key or loop_)
            all_tokens: list[str] = []
            while i < len(lines):
                row_raw = lines[i]
                row = row_raw.strip()
                if not row or row.startswith('#'):
                    i += 1
                    continue
                if row.startswith('_') or row == 'loop_' or row.lower().startswith('data_'):
                    break
                # Handle semicolon text blocks:
                # Opening ';' is a standalone semicolon line. Closing ';' may have
                # additional tokens after it (e.g. "; 1982 38 1451 ...").
                if row == ';':
                    # Opening: skip text until closing ';' (line starting with ';')
                    block_lines: list[str] = []
                    i += 1
                    while i < len(lines):
                        bl = lines[i]
                        if bl[:1] == ';':        # closing ';', first char of line
                            remainder = bl[1:].strip()
                            if remainder:
                                all_tokens.extend(_tokenize(remainder))
                            i += 1
                            break
                        block_lines.append(bl.rstrip())
                        i += 1
                    all_tokens.append('\n'.join(block_lines))
                    continue
                all_tokens.extend(_tokenize(row))
                i += 1

            # Group tokens into logical rows of len(columns) tokens each
            n_cols = len(columns)
            rows_tokens: list[list[str]] = []
            if n_cols > 0:
                for j in range(0, len(all_tokens) - n_cols + 1, n_cols):
                    rows_tokens.append(all_tokens[j:j + n_cols])

            loops.append({'columns': columns, 'rows_tokens': rows_tokens})
            continue

        # ---- key-value pair ----
        if line.startswith('_'):
            # Inline: "_key value" on same line
            sp = line.find(' ')
            if sp > 0:
                key = line[:sp]
                val = line[sp:].strip()
                kv[key] = _strip_quotes(val)
                i += 1
            else:
                # Value on next line
                key = line
                i += 1
                if i < len(lines):
                    val_line = lines[i].strip()
                    if val_line == ';':
                        # Multi-line semicolon block — closing ';' may have extra content
                        i += 1
                        parts: list[str] = []
                        while i < len(lines):
                            if lines[i][:1] == ';':   # closing ';' (first char of line)
                                i += 1
                                break
                            parts.append(lines[i])
                            i += 1
                        kv[key] = '\n'.join(parts)
                    else:
                        kv[key] = _strip_quotes(val_line)
                        i += 1
            continue

        i += 1

    # --------------------------------------------------------------------------
    # Extract lattice parameters
    # --------------------------------------------------------------------------
    def _kv_float(keys, default: float) -> float:
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            v = kv.get(k, '').strip()
            if v and v not in ('?', '.'):
                try:
                    return _parse_cif_float(v)
                except ValueError:
                    pass
        return default

    a = _kv_float('_cell_length_a', 1.0)
    b = _kv_float('_cell_length_b', a)
    c = _kv_float('_cell_length_c', a)
    alpha = _kv_float('_cell_angle_alpha', 90.0)
    beta  = _kv_float('_cell_angle_beta',  90.0)
    gamma = _kv_float('_cell_angle_gamma', 90.0)

    # --------------------------------------------------------------------------
    # Extract space group
    # --------------------------------------------------------------------------
    sg_number: int | None = None
    for k in ('_space_group_IT_number',
              '_symmetry_Int_Tables_number',
              '_space_group_it_number'):
        v = kv.get(k, '').strip()
        if v and v not in ('?', '.'):
            try:
                sg_number = int(float(v))
                break
            except ValueError:
                pass

    sg_symbol: str | None = None
    for k in ('_space_group_name_H-M_alt',
              '_symmetry_space_group_name_H-M',
              '_space_group_name_H_M_alt',
              '_space_group_name_h-m_alt'):
        v = kv.get(k, '').strip()
        if v and v not in ('?', '.'):
            sg_symbol = _strip_quotes(v)
            break

    # --------------------------------------------------------------------------
    # Extract symmetry operations
    # --------------------------------------------------------------------------
    _SYM_OP_KEYS = (
        '_space_group_symop_operation_xyz',
        '_symmetry_equiv_pos_as_xyz',
        '_space_group_symop.operation_xyz',
    )
    sym_ops: list[str] = []
    for loop in loops:
        cols = loop['columns']
        idx = None
        for k in _SYM_OP_KEYS:
            if k in cols:
                idx = cols.index(k)
                break
        if idx is None:
            continue
        for row_toks in loop['rows_tokens']:
            if idx < len(row_toks):
                op = row_toks[idx].replace("'", "").strip()
                if op:
                    sym_ops.append(op)
        if sym_ops:
            break   # use first matching loop

    if not sym_ops:
        sym_ops = ['x,y,z']

    # --------------------------------------------------------------------------
    # Extract atom sites (asymmetric unit)
    # --------------------------------------------------------------------------
    _LABEL_KEYS  = ('_atom_site_label',        '_atom_site.label')
    _TYPE_KEYS   = ('_atom_site_type_symbol',   '_atom_site.type_symbol')
    _FX_KEYS     = ('_atom_site_fract_x',       '_atom_site.fract_x')
    _FY_KEYS     = ('_atom_site_fract_y',       '_atom_site.fract_y')
    _FZ_KEYS     = ('_atom_site_fract_z',       '_atom_site.fract_z')
    _OCC_KEYS    = ('_atom_site_occupancy',     '_atom_site.occupancy')
    _UISO_KEYS   = ('_atom_site_U_iso_or_equiv', '_atom_site.U_iso_or_equiv', '_atom_site_u_iso_or_equiv')
    _BISO_KEYS   = ('_atom_site_B_iso_or_equiv', '_atom_site.B_iso_or_equiv', '_atom_site_b_iso_or_equiv')

    asym_atoms: list[dict] = []

    for loop in loops:
        cols = loop['columns']

        def _find_col(keys):
            for k in keys:
                if k in cols:
                    return cols.index(k)
            return None

        i_label = _find_col(_LABEL_KEYS)
        i_type  = _find_col(_TYPE_KEYS)
        i_fx    = _find_col(_FX_KEYS)
        i_fy    = _find_col(_FY_KEYS)
        i_fz    = _find_col(_FZ_KEYS)
        i_occ   = _find_col(_OCC_KEYS)
        i_uiso  = _find_col(_UISO_KEYS)
        i_biso  = _find_col(_BISO_KEYS)

        if i_label is None or i_fx is None:
            continue   # not an atom site loop

        for row_toks in loop['rows_tokens']:
            if len(row_toks) <= i_fx:
                continue

            label = row_toks[i_label] if i_label < len(row_toks) else '?'

            # Determine element symbol
            if i_type is not None and i_type < len(row_toks):
                raw_sym = row_toks[i_type]
            else:
                # Parse element from label (e.g. "Fe1" → "Fe", "O2" → "O")
                m = re.match(r'^([A-Z][a-z]?)', label)
                raw_sym = m.group(1) if m else 'X'

            # Strip charge/oxidation state notation: Fe2+ → Fe, O2- → O
            m = re.match(r'^([A-Za-z]+)', raw_sym)
            raw_sym = m.group(1) if m else 'X'
            elem = raw_sym[:1].upper() + raw_sym[1:].lower() if len(raw_sym) > 1 else raw_sym.upper()

            # Fractional coordinates
            try:
                fx = _parse_cif_float(row_toks[i_fx]) if i_fx < len(row_toks) else 0.0
                fy = _parse_cif_float(row_toks[i_fy]) if (i_fy is not None and i_fy < len(row_toks)) else 0.0
                fz = _parse_cif_float(row_toks[i_fz]) if (i_fz is not None and i_fz < len(row_toks)) else 0.0
            except (ValueError, TypeError):
                continue

            # Occupancy
            occ = 1.0
            if i_occ is not None and i_occ < len(row_toks):
                try:
                    occ = _parse_cif_float(row_toks[i_occ])
                except (ValueError, TypeError):
                    occ = 1.0

            # Isotropic displacement (U in Å²; B = 8π² U)
            uiso = np.nan
            for i_col, scale in ((i_uiso, 1.0), (i_biso, 1.0 / (8.0 * np.pi ** 2))):
                if i_col is not None and i_col < len(row_toks) and np.isnan(uiso):
                    try:
                        uiso = _parse_cif_float(row_toks[i_col]) * scale
                    except (ValueError, TypeError):
                        pass

            asym_atoms.append({'sym': elem, 'fx': fx, 'fy': fy, 'fz': fz, 'occ': occ, 'uiso': uiso,
                               'label': label})

        if asym_atoms:
            break   # use first matching atom-site loop

    # --------------------------------------------------------------------------
    # Anisotropic displacements -> U_eq (for sites without an isotropic value)
    # --------------------------------------------------------------------------
    aniso = _parse_aniso(loops)
    if aniso:
        ueq = {lab: _u_equivalent(u, (a, b, c, alpha, beta, gamma)) for lab, u in aniso.items()}
        for atom in asym_atoms:
            if np.isnan(atom['uiso']) and atom['label'] in ueq:
                atom['uiso'] = ueq[atom['label']]

    # --------------------------------------------------------------------------
    # Symmetry operations from the space group, if the CIF lists none
    # --------------------------------------------------------------------------
    if sym_ops == ['x,y,z'] and (sg_symbol or sg_number):
        sym_ops = _ops_from_space_group(sg_symbol, sg_number) or sym_ops

    # --------------------------------------------------------------------------
    # Expand asymmetric unit via symmetry operations
    # --------------------------------------------------------------------------
    basis_list, elements_list, occ_list, uiso_list, label_list = expand_asymmetric_unit(asym_atoms, sym_ops)

    # --------------------------------------------------------------------------
    # Material name
    # --------------------------------------------------------------------------
    name = (
        kv.get('_chemical_name_mineral') or
        kv.get('_chemical_name_common') or
        kv.get('_chemical_formula_sum') or
        Path(path).stem
    )
    name = _strip_quotes(str(name)).strip()

    return {
        'name':               name,
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'space_group_number': sg_number,
        'space_group_symbol': sg_symbol,
        'sym_ops':            sym_ops,
        'basis':              np.array(basis_list, dtype=float) if basis_list else np.zeros((0, 3)),
        'elements':           elements_list,
        'occupancies':        np.array(occ_list, dtype=float) if occ_list else np.zeros(0),
        'uiso':               np.array(uiso_list, dtype=float) if uiso_list else np.zeros(0),
        'labels':             label_list,
    }
