"""
Free memory of an OpenCL device, where the driver lets us know it.

OpenCL itself only reports the total (CL_DEVICE_GLOBAL_MEM_SIZE). On a shared GPU most of that
may be in use by other processes, so budgets computed from the total can be far too large. For
NVIDIA devices the free memory comes from NVML (libnvidia-ml, part of the driver), matched to the
OpenCL device by its PCI bus (cl_nv_device_attribute_query); for AMD from
CL_DEVICE_GLOBAL_FREE_MEMORY_AMD. Anything else, or any failure: None (callers then fall back to
the total).
"""
from __future__ import annotations

import ctypes

_NVML = None  # the loaded and initialised library, False if unavailable


class _MemInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _PciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def _nvml():
    global _NVML
    if _NVML is None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            _NVML = lib if lib.nvmlInit_v2() == 0 else False
        except OSError:
            _NVML = False
    return _NVML or None


def _nvidia_free(device):
    lib = _nvml()
    if lib is None:
        return None
    try:
        bus = int(device.pci_bus_id_nv)
        slot = int(device.pci_slot_id_nv)
        domain = int(getattr(device, "pci_domain_id_nv", 0))
    except Exception:
        return None
    n = ctypes.c_uint()
    if lib.nvmlDeviceGetCount_v2(ctypes.byref(n)) != 0:
        return None
    matches = []
    for i in range(n.value):
        h = ctypes.c_void_p()
        pci = _PciInfo()
        if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0 or \
                lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(pci)) != 0:
            continue
        if pci.bus == bus and pci.domain == domain:
            matches.append((pci.device, h))
    if len(matches) > 1:  # the slot id is the PCI device, possibly with the function in its low 3 bits
        matches = [m for m in matches if m[0] in (slot, slot >> 3)]
    if len(matches) != 1:
        return None
    mem = _MemInfo()
    if lib.nvmlDeviceGetMemoryInfo(matches[0][1], ctypes.byref(mem)) != 0:
        return None
    return int(mem.free)


def free_device_bytes(device):
    """Free memory of a pyopencl device in bytes, or None when the driver does not tell."""
    try:
        vendor = device.vendor.lower()
    except Exception:
        return None
    if "nvidia" in vendor:
        return _nvidia_free(device)
    if "advanced micro devices" in vendor or "amd" in vendor:
        try:
            return int(device.global_free_memory_amd[0]) * 1024
        except Exception:
            return None
    return None
