"""
Space group number <-> short Hermann-Mauguin symbol lookup table, and the
integral (lattice-centering) reflection conditions derived from it.
 
Source: IUCr space-group numbering (International Tables for Crystallography,
Vol. A), as tabulated at:
    https://en.wikipedia.org/wiki/List_of_space_groups
(the seven sub-tables: List of triclinic / monoclinic / orthorhombic /
tetragonal / trigonal / hexagonal / cubic space groups)
"""

import numpy as np

SPACE_GROUP_SYMBOLS = {
    1: "P1", 2: "P-1",
    3: "P2", 4: "P21", 5: "C2",
    6: "Pm", 7: "Pc", 8: "Cm", 9: "Cc",
    10: "P2/m", 11: "P21/m", 12: "C2/m", 13: "P2/c", 14: "P21/c", 15: "C2/c",
    16: "P222", 17: "P2221", 18: "P21212", 19: "P212121", 20: "C2221",
    21: "C222", 22: "F222", 23: "I222", 24: "I212121",
    25: "Pmm2", 26: "Pmc21", 27: "Pcc2", 28: "Pma2", 29: "Pca21", 30: "Pnc2",
    31: "Pmn21", 32: "Pba2", 33: "Pna21", 34: "Pnn2", 35: "Cmm2", 36: "Cmc21",
    37: "Ccc2", 38: "Amm2", 39: "Aem2", 40: "Ama2", 41: "Aea2", 42: "Fmm2",
    43: "Fdd2", 44: "Imm2", 45: "Iba2", 46: "Ima2",
    47: "Pmmm", 48: "Pnnn", 49: "Pccm", 50: "Pban", 51: "Pmma", 52: "Pnna",
    53: "Pmna", 54: "Pcca", 55: "Pbam", 56: "Pccn", 57: "Pbcm", 58: "Pnnm",
    59: "Pmmn", 60: "Pbcn", 61: "Pbca", 62: "Pnma", 63: "Cmcm", 64: "Cmce",
    65: "Cmmm", 66: "Cccm", 67: "Cmme", 68: "Ccce", 69: "Fmmm", 70: "Fddd",
    71: "Immm", 72: "Ibam", 73: "Ibca", 74: "Imma",
    75: "P4", 76: "P41", 77: "P42", 78: "P43", 79: "I4", 80: "I41",
    81: "P-4", 82: "I-4",
    83: "P4/m", 84: "P42/m", 85: "P4/n", 86: "P42/n", 87: "I4/m", 88: "I41/a",
    89: "P422", 90: "P4212", 91: "P4122", 92: "P41212", 93: "P4222",
    94: "P42212", 95: "P4322", 96: "P43212", 97: "I422", 98: "I4122",
    99: "P4mm", 100: "P4bm", 101: "P42cm", 102: "P42nm", 103: "P4cc",
    104: "P4nc", 105: "P42mc", 106: "P42bc", 107: "I4mm", 108: "I4cm",
    109: "I41md", 110: "I41cd",
    111: "P-42m", 112: "P-42c", 113: "P-421m", 114: "P-421c",
    115: "P-4m2", 116: "P-4c2", 117: "P-4b2", 118: "P-4n2",
    119: "I-4m2", 120: "I-4c2", 121: "I-42m", 122: "I-42d",
    123: "P4/mmm", 124: "P4/mcc", 125: "P4/nbm", 126: "P4/nnc",
    127: "P4/mbm", 128: "P4/mnc", 129: "P4/nmm", 130: "P4/ncc",
    131: "P42/mmc", 132: "P42/mcm", 133: "P42/nbc", 134: "P42/nnm",
    135: "P42/mbc", 136: "P42/mnm", 137: "P42/nmc", 138: "P42/ncm",
    139: "I4/mmm", 140: "I4/mcm", 141: "I41/amd", 142: "I41/acd",
    143: "P3", 144: "P31", 145: "P32", 146: "R3",
    147: "P-3", 148: "R-3",
    149: "P312", 150: "P321", 151: "P3112", 152: "P3121", 153: "P3212",
    154: "P3221", 155: "R32",
    156: "P3m1", 157: "P31m", 158: "P3c1", 159: "P31c", 160: "R3m", 161: "R3c",
    162: "P-31m", 163: "P-31c", 164: "P-3m1", 165: "P-3c1",
    166: "R-3m", 167: "R-3c",
    168: "P6", 169: "P61", 170: "P65", 171: "P62", 172: "P64", 173: "P63",
    174: "P-6",
    175: "P6/m", 176: "P63/m",
    177: "P622", 178: "P6122", 179: "P6522", 180: "P6222", 181: "P6422",
    182: "P6322",
    183: "P6mm", 184: "P6cc", 185: "P63cm", 186: "P63mc",
    187: "P-6m2", 188: "P-6c2", 189: "P-62m", 190: "P-62c",
    191: "P6/mmm", 192: "P6/mcc", 193: "P63/mcm", 194: "P63/mmc",
    195: "P23", 196: "F23", 197: "I23", 198: "P213", 199: "I213",
    200: "Pm-3", 201: "Pn-3", 202: "Fm-3", 203: "Fd-3", 204: "Im-3",
    205: "Pa-3", 206: "Ia-3",
    207: "P432", 208: "P4232", 209: "F432", 210: "F4132", 211: "I432",
    212: "P4332", 213: "P4132", 214: "I4132",
    215: "P-43m", 216: "F-43m", 217: "I-43m", 218: "P-43n", 219: "F-43c",
    220: "I-43d",
    221: "Pm-3m", 222: "Pn-3n", 223: "Pm-3n", 224: "Pn-3m",
    225: "Fm-3m", 226: "Fm-3c", 227: "Fd-3m", 228: "Fd-3c",
    229: "Im-3m", 230: "Ia-3d",
}




# symbol -> number (for the reverse lookup); assumes SPACE_GROUP_SYMBOLS
# keys are unique, which they are for the standard setting used here.
SPACE_GROUP_NUMBERS = {sym: num for num, sym in SPACE_GROUP_SYMBOLS.items()}
 
# Centering letter is always the first character of the short symbol.
VALID_CENTERINGS = {"P", "A", "B", "C", "I", "F", "R"}
 
 
def resolve_space_group(number=None, symbol=None):
    """
    Given a space group number and/or symbol, return (number, symbol),
    filling in whichever is missing from the lookup table. Raises if both
    are given and disagree, or if neither is given, or if the number/symbol
    isn't found.
    """
    if number is None and symbol is None:
        raise ValueError("Provide a space group 'number' or 'symbol'.")
 
    if number is not None:
        if number not in SPACE_GROUP_SYMBOLS:
            raise ValueError(f"Unknown space group number: {number}")
        looked_up_symbol = SPACE_GROUP_SYMBOLS[number]
        if symbol is not None and symbol != looked_up_symbol:
            raise ValueError(
                f"space_group_number={number} implies symbol "
                f"'{looked_up_symbol}', but symbol='{symbol}' was also given."
            )
        return number, looked_up_symbol
 
    # only symbol given
    if symbol not in SPACE_GROUP_NUMBERS:
        raise ValueError(
            f"Unknown space group symbol: {symbol!r}. Note this lookup "
            "expects the standard-setting short symbol, e.g. 'Fm-3m'."
        )
    return SPACE_GROUP_NUMBERS[symbol], symbol
 
 
def centering_of(symbol):
    """Extract the centering letter (P/A/B/C/I/F/R) from a short symbol."""
    letter = symbol[0]
    if letter not in VALID_CENTERINGS:
        raise ValueError(f"Unrecognised centering letter {letter!r} in {symbol!r}")
    return letter
 
 
def satisfies_centering_condition(hkl, centering, rhombohedral_axes="hexagonal"):
    """
    Integral (centering) reflection condition. hkl: (N, 3) integer array.
    Returns a boolean mask, True = reflection is allowed (not extinct).
 
    R-centered groups are ambiguous between rhombohedral axes (no condition)
    and hexagonal axes, obverse setting (-h+k+l == 0 mod 3, the far more
    common convention and the one used here by default). If your R-group
    material is actually indexed on rhombohedral axes, pass
    rhombohedral_axes='rhombohedral'.
    """
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
 
    if centering == "P":
        return np.ones(len(hkl), dtype=bool)
    elif centering == "I":
        return (h + k + l) % 2 == 0
    elif centering == "F":
        return (h % 2 == k % 2) & (k % 2 == l % 2)
    elif centering == "A":
        return (k + l) % 2 == 0
    elif centering == "B":
        return (h + l) % 2 == 0
    elif centering == "C":
        return (h + k) % 2 == 0
    elif centering == "R":
        if rhombohedral_axes == "rhombohedral":
            return np.ones(len(hkl), dtype=bool)
        elif rhombohedral_axes == "hexagonal":
            return (-h + k + l) % 3 == 0
        else:
            raise ValueError(
                "rhombohedral_axes must be 'hexagonal' or 'rhombohedral', "
                f"got {rhombohedral_axes!r}"
            )
    else:
        raise ValueError(f"Unhandled centering: {centering!r}")
