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

            asym_atoms.append({'sym': elem, 'fx': fx, 'fy': fy, 'fz': fz, 'occ': occ})

        if asym_atoms:
            break   # use first matching atom-site loop

    # --------------------------------------------------------------------------
    # Expand asymmetric unit via symmetry operations
    # --------------------------------------------------------------------------
    parsed_ops = [_parse_sym_op(op) for op in sym_ops]

    basis_list:    list[list[float]] = []
    elements_list: list[str]         = []
    occ_list:      list[float]       = []

    for atom in asym_atoms:
        for op in parsed_ops:
            pos = _apply_sym_op(op, atom['fx'], atom['fy'], atom['fz'])
            mx = _mod1(pos[0])
            my = _mod1(pos[1])
            mz = _mod1(pos[2])

            is_dup = any(
                _close_modulo(b[0], mx) and _close_modulo(b[1], my) and _close_modulo(b[2], mz)
                for b in basis_list
            )
            if not is_dup:
                basis_list.append([mx, my, mz])
                elements_list.append(atom['sym'])
                occ_list.append(atom['occ'])

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
    }
