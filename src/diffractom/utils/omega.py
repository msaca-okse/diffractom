"""The rotation angles of a scan as the operators use them.

Every operator takes ``cfg["angle_range"] = [a0, a1]`` (degrees) as the EDGES of the rotation that the N_Omega data
bins cover together: bin i is the interval [a0 + i d, a0 + (i + 1) d], d = (a1 - a0) / N_Omega. The diffraction part
of the model integrates over each bin's interval (the omega quadrature), and the tomographic projection of bin i is
taken at its centre a0 + (i + 1/2) d. The orientations are reconstructed in the sample frame defined by these
angles, so an offset of the angles is an offset of every reconstructed orientation about the rotation axis: if the
recorded angle of each frame is the start (or the end) of its exposure interval and is used as if it were the
centre, all orientations turn by half a frame.

Use :func:`angle_range_from_frames` to build ``angle_range`` from the angles recorded with the frames; it needs the
labelling convention of the recorded angles explicitly. Examples: DanMAX continuous scans (Al1050, 2025) record
``tom_ry`` = the encoder minimum over the exposure (``pcap_rot_min``), i.e. the start of each frame's interval
for an increasing rotation; the simulated scans of the companion repository store the start of [omega, omega + step].
"""
from __future__ import annotations

import numpy as np

LABELS = ("start", "center", "end")


def angle_range_from_frames(omega, label: str, frames_per_bin: int = 1, n_frames: int | None = None,
                            rtol: float = 0.05):
    """``cfg["angle_range"]`` (degrees) and the bin centres for data binned from frames with recorded angles.

    omega : (n,) recorded rotation angle of every frame (degrees), equally spaced, increasing or decreasing; a
        wrap at +-360 (encoder readback) is unwrapped.
    label : "start", "center" or "end": which point of the frame's rotation interval the recorded angle is (for
        the direction of the rotation: "start" is where the exposure begins). No default: state it.
    frames_per_bin : frames summed into one data bin (consecutive, starting at frame 0).
    n_frames : frames used (default: the largest multiple of frames_per_bin <= len(omega)); e.g. 3000 of a
        3001-point continuous scan whose last point repeats the first interval.
    rtol : allowed relative deviation of the steps from their median.

    Returns (angle_range [a0, a1], centres (N_Omega,)), a0 / a1 the outer edges of the first / last bin.
    """
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS} (where in its rotation interval a frame's angle was "
                         f"recorded), not {label!r}")
    om = np.unwrap(np.deg2rad(np.asarray(omega, np.float64)))
    om = np.rad2deg(om)
    if om.ndim != 1 or len(om) < 2:
        raise ValueError("omega: at least two frame angles")
    steps = np.diff(om)
    step = float(np.median(steps))
    if step == 0 or np.any(np.abs(steps - step) > rtol * abs(step)):
        raise ValueError(f"omega: the steps are not uniform (median {step:g}, range {steps.min():g} .. "
                         f"{steps.max():g} deg)")
    fpb = int(frames_per_bin)
    if n_frames is None:
        n_frames = len(om) // fpb * fpb
    if n_frames % fpb or n_frames > len(om) or n_frames < fpb:
        raise ValueError("n_frames must be a positive multiple of frames_per_bin, at most len(omega)")
    shift = {"start": 0.0, "center": 0.5, "end": 1.0}[label]
    # the interval of frame j is [s_j, s_j + step] (s_j its start, in the direction of the rotation); fit s_0 to all
    # frames (robust to readback jitter)
    j = np.arange(n_frames)
    s0 = float(np.mean(om[:n_frames] - shift * step - j * step))
    a0, a1 = s0, s0 + n_frames * step
    n_bins = n_frames // fpb
    d = (a1 - a0) / n_bins
    centres = a0 + (np.arange(n_bins) + 0.5) * d
    return [a0, a1], centres
