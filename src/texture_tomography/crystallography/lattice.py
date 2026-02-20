import numpy as np

"""
This file defines lattice and reciprocal-lattice matrices for the seven
Bravais lattice systems. Each function returns the direct lattice matrix A
and its reciprocal lattice matrix B = 2π (A⁻¹)ᵀ.

A is built from the 6 lattice parameters:
    a, b, c, α, β, γ
where angles are given in degrees.

Use:
    A, B = tetragonal(a=3.2, c=5.1)
    h_vec = B @ np.array([h, k, l])
"""

def reciprocal_lattice(A):
    """Return reciprocal lattice matrix B = 2π (A⁻¹)ᵀ"""
    return 2 * np.pi * np.linalg.inv(A).T


def cubic(a=1):
    A = np.array([[a, 0, 0],
                  [0, a, 0],
                  [0, 0, a]]).T
    return A, reciprocal_lattice(A)


def orthorhombic(a=1, b=2, c=3):
    A = np.array([[a, 0, 0],
                  [0, b, 0],
                  [0, 0, c]]).T
    return A, reciprocal_lattice(A)


def tetragonal(a=1, c=1.5):
    A = np.array([[a, 0, 0],
                  [0, a, 0],
                  [0, 0, c]]).T
    return A, reciprocal_lattice(A)


def hexagonal(a=1, c=1.633):
    A = np.array([[a, 0, 0],
                  [-0.5 * a, np.sqrt(3) / 2 * a, 0],
                  [0, 0, c]]).T
    return A, reciprocal_lattice(A)


def trigonal_rhombohedral(a=1, alpha=60):
    """Rhombohedral lattice (a = b = c, α = β = γ != 90°)"""
    alpha_r = np.deg2rad(alpha)
    A = np.array([
        [a, 0, 0],
        [a * (np.cos(alpha_r)), a * np.sin(alpha_r), 0],
        [a * (np.cos(alpha_r)),
         a * (np.cos(alpha_r) - np.cos(alpha_r)**2) / np.sin(alpha_r),
         a * np.sqrt(1 - 3 * (np.cos(alpha_r)**2) + 2 * (np.cos(alpha_r)**3)) / np.sin(alpha_r)]
    ]).T
    return A, reciprocal_lattice(A)


def monoclinic(a=1, b=1.2, c=1.5, beta=110):
    """Monoclinic lattice with unique b-axis (β ≠ 90° between a and c)."""
    beta_r = np.deg2rad(beta)
    A = np.array([
        [a, 0, c * np.cos(beta_r)],
        [0, b, 0],
        [0, 0, c * np.sin(beta_r)]  # c lies in a–c plane
    ]).T
    return A, reciprocal_lattice(A)


def triclinic(a=1, b=1.1, c=1.2, alpha=80, beta=100, gamma=110):
    """General triclinic lattice (no 90° angles)."""
    alpha_r = np.deg2rad(alpha)
    beta_r = np.deg2rad(beta)
    gamma_r = np.deg2rad(gamma)

    # Compute the third lattice vector in 3D Cartesian coordinates
    A = np.zeros((3, 3))
    A[:, 0] = [a, 0, 0]
    A[:, 1] = [b * np.cos(gamma_r), b * np.sin(gamma_r), 0]
    A[:, 2] = [
        c * np.cos(beta_r),
        c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r),
        c * np.sqrt(
            1
            - np.cos(alpha_r) ** 2
            - np.cos(beta_r) ** 2
            - np.cos(gamma_r) ** 2
            + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
        ) / np.sin(gamma_r)
    ]
    return A, reciprocal_lattice(A)
