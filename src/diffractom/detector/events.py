"""The measured counts as a list of events: (frame, cell, count) for every cell with a non-zero count.

A frame is one (rotation bin, translation) of the scan: frame = i_omega * My + i_y, the row of the operator's data
array. Raw frames are summed into the operator's rotation bins (frames whose rotation intervals make up one bin; the
model of a bin is the integral over it, so the sum is the statistic the model describes), and pixels into cells
(cells.py). Cells with zero counts are not stored: their contribution to the Poisson likelihood is the model alone,
summed over all cells of all frames through the sensitivity (raw_model.py). For low-count, spotty data the list is
much smaller than the frames; for frames dominated by background (most pixels counted) use larger cells.
"""
import numpy as np


class EventList:
    """frame (E,) int32, cell (E,) int32, count (E,) float32; n_omega, n_y; exposure (n_omega, n_y) float32: the
    relative exposure of every frame (e.g. the number of raw frames summed, or the beam monitor), default 1.
    unit: "cells" (cell = a DetectorCells cell) or "pixels" (cell = a used pixel of DetectorPixels: the raw data
    before any model choice; to_cells maps them onto the cells of a model)."""

    def __init__(self, frame, cell, count, n_omega, n_y, n_cells, exposure=None, unit="cells"):
        self.frame = np.ascontiguousarray(frame, dtype=np.int32)
        self.cell = np.ascontiguousarray(cell, dtype=np.int32)
        self.count = np.ascontiguousarray(count, dtype=np.float32)
        if not (self.frame.shape == self.cell.shape == self.count.shape) or self.frame.ndim != 1:
            raise ValueError("frame, cell and count must be 1-D arrays of the same length")
        self.n_omega, self.n_y, self.n_cells = int(n_omega), int(n_y), int(n_cells)
        if len(self.frame) and (self.frame.min() < 0 or self.frame.max() >= self.n_omega * self.n_y):
            raise ValueError("frame numbers outside 0 .. n_omega * n_y - 1")
        if len(self.cell) and (self.cell.min() < 0 or self.cell.max() >= self.n_cells):
            raise ValueError("cell numbers outside 0 .. n_cells - 1")
        self.exposure = (np.ones((self.n_omega, self.n_y), np.float32) if exposure is None
                         else np.ascontiguousarray(exposure, dtype=np.float32).reshape(self.n_omega, self.n_y))
        if unit not in ("cells", "pixels"):
            raise ValueError("unit must be 'cells' or 'pixels'")
        self.unit = unit

    def __len__(self):
        return len(self.count)

    @property
    def total(self):
        return float(self.count.astype(np.float64).sum())

    def save(self, path):
        np.savez(path, frame=self.frame, cell=self.cell, count=self.count, exposure=self.exposure,
                 shape=np.array([self.n_omega, self.n_y, self.n_cells]), unit=np.array(self.unit))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n_omega, n_y, n_cells = (int(v) for v in d["shape"])
        unit = str(d["unit"]) if "unit" in d.files else "cells"
        return cls(d["frame"], d["cell"], d["count"], n_omega, n_y, n_cells, d["exposure"], unit)

    @classmethod
    def builder(cls, cells_or_pixels, n_omega, n_y):
        return EventBuilder(cells_or_pixels, n_omega, n_y)

    def unit_sums(self):
        """(n_cells,) counts of every cell (or pixel) summed over all frames, e.g. the summed pattern for
        RadialProfiles.from_powder."""
        return np.bincount(self.cell, weights=self.count.astype(np.float64), minlength=self.n_cells)

    def to_cells(self, cells):
        """Pixel events (unit "pixels", the pixels of cells.pixels) -> the events of the model's cells: counts of
        pixels in one cell and frame are summed, pixels that are not modelled are dropped (dropped_counts)."""
        if self.unit != "pixels":
            raise ValueError("to_cells needs pixel events")
        if self.n_cells != cells.pixels.n_pixels:
            raise ValueError("the events' pixels are not the cells' pixels")
        c = cells.pixel_cell[self.cell]
        ok = c >= 0
        key = self.frame[ok].astype(np.int64) * cells.n_cells + c[ok]
        order = np.argsort(key, kind="stable")
        key = key[order]
        ct = self.count[ok][order].astype(np.float64)
        uk, start = np.unique(key, return_index=True)
        sums = np.add.reduceat(ct, start) if len(ct) else ct
        out = EventList((uk // cells.n_cells).astype(np.int32), (uk % cells.n_cells).astype(np.int32),
                        sums.astype(np.float32), self.n_omega, self.n_y, cells.n_cells, self.exposure)
        out.dropped_counts = float(self.count[~ok].astype(np.float64).sum())
        return out


class EventBuilder:
    """Collects frames into an EventList. Every add_* call adds the counts of (part of) one raw frame to the frame
    bin (i_omega, i_y); several raw frames may go into one bin (they are summed). Built on DetectorPixels, the events
    are pixel events (to_cells maps them onto a model's cells later); on DetectorCells, cell events."""

    def __init__(self, cells, n_omega, n_y):
        if hasattr(cells, "pixel_cell"):
            self.lut = cells.frame_cell_lookup()
            self.n_cells = cells.n_cells
            self.unit = "cells"
        else:
            self.lut = cells.frame_lookup()
            self.n_cells = cells.n_pixels
            self.unit = "pixels"
        self.n_omega, self.n_y = int(n_omega), int(n_y)
        self._parts = []
        self.dropped = 0.0          # counts on pixels that are not modelled or not finite
        self.n_nonfinite = 0
        self.exposure = np.zeros((self.n_omega, self.n_y), np.float64)

    def add_sparse(self, i_omega, i_y, flat_index, values, exposure=1.0):
        """Counts of a raw frame at the frame pixels flat_index (row * n_cols + col)."""
        cell = self.lut[np.asarray(flat_index, dtype=np.int64)]
        v = np.asarray(values, dtype=np.float64)
        fin = np.isfinite(v)
        self.n_nonfinite += int((~fin).sum())
        ok = (cell >= 0) & fin & (v != 0)
        self.dropped += float(v[fin & (cell < 0)].sum())
        self._add(i_omega, i_y, cell[ok], v[ok])
        self.exposure[i_omega, i_y] += exposure

    def add_dense(self, i_omega, i_y, frame, exposure=1.0):
        """A dense raw frame (all pixels)."""
        f = np.asarray(frame).ravel()
        nz = np.flatnonzero(f)
        self.add_sparse(i_omega, i_y, nz, f[nz], exposure)

    def _add(self, i_omega, i_y, cell, v):
        if len(cell) == 0:
            return
        # sum several pixels of one cell (cells of several pixels)
        u, inv = np.unique(cell, return_inverse=True)
        s = np.bincount(inv, weights=v)
        fr = np.full(len(u), int(i_omega) * self.n_y + int(i_y), dtype=np.int32)
        self._parts.append((fr, u.astype(np.int32), s.astype(np.float32)))

    def finish(self, exposure=None):
        """The EventList, events sorted by frame, then cell; several additions to one (frame, cell) are summed.
        exposure: overrides the summed exposures of the add_* calls."""
        if self._parts:
            fr = np.concatenate([p[0] for p in self._parts])
            ce = np.concatenate([p[1] for p in self._parts])
            ct = np.concatenate([p[2] for p in self._parts]).astype(np.float64)
        else:
            fr = ce = np.zeros(0, np.int32)
            ct = np.zeros(0)
        key = fr.astype(np.int64) * self.n_cells + ce
        order = np.argsort(key, kind="stable")
        key, ct = key[order], ct[order]
        uk, start = np.unique(key, return_index=True)
        sums = np.add.reduceat(ct, start) if len(ct) else ct
        exp = self.exposure if exposure is None else exposure
        return EventList((uk // self.n_cells).astype(np.int32), (uk % self.n_cells).astype(np.int32),
                         sums.astype(np.float32), self.n_omega, self.n_y, self.n_cells, exp, self.unit)
