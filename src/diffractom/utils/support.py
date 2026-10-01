"""Support masks for the reconstructed image."""
import numpy as np


def fov_support_mask(Nx, Ny, n_detectors, angles=None, image_width=None, detector_width=None,
                     detector_shift=0.0):
    """
    Pixels inside the field of view at every projection angle (the parallel-beam geometry
    of ParallelRadon and gratopy).

    A sample that stays in the beam during the whole scan lies inside this region, which
    depends on the detector and the angles, not on the reconstruction grid. The rotation
    axis is at the image centre and the detector centre is ``detector_shift`` from it along
    the detector line. For a full rotation (or no shift) the region is the disk
    ``r <= detector_width / 2 - |detector_shift|``; with a shift and a half rotation it is
    larger, since each point is only seen from one side.

    Parameters
    ----------
    Nx, Ny : int
        Image shape.
    n_detectors : int
        Number of detector bins.
    angles : (R,) array, radians, optional
        Projection angles. None gives the disk (seen at every angle of a full rotation).
    image_width, detector_width : float, optional
        Physical widths of the image and the detector line (default Nx and n_detectors,
        i.e. pixels and bins of unit size, as in the forward operators).
    detector_shift : float
        Centre-of-rotation offset, in the units of detector_width.

    Returns
    -------
    (Ny, Nx) bool array, mask[iy, ix] (the layout of a coefficient image), True where the pixel
    centre projects onto the detector at every angle.
    """
    image_width = float(Nx if image_width is None else image_width)
    detector_width = float(n_detectors if detector_width is None else detector_width)
    delta_x = image_width / max(Nx, Ny)
    x = (np.arange(Nx) - (Nx - 1) / 2.0)[None, :] * delta_x
    y = (np.arange(Ny) - (Ny - 1) / 2.0)[:, None] * delta_x
    half = detector_width / 2.0
    if angles is None:
        return x ** 2 + y ** 2 <= (half - abs(float(detector_shift))) ** 2

    mask = np.ones((Ny, Nx), dtype=bool)
    for a in np.asarray(angles, dtype=np.float64):
        # detector coordinate of the pixel centre relative to the detector centre (ParallelRadon)
        t = np.cos(a - 0.5 * np.pi) * x + np.sin(a - 0.5 * np.pi) * y - detector_shift
        mask &= np.abs(t) <= half + 1e-9 * detector_width
    return mask
