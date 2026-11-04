"""Shared test set-up: release the OpenCL context of every finished test. Each test creates its own context (about
300 MB of device memory on NVIDIA), and pyopencl's per-context kernel caches (pyopencl.array operations) keep it
alive, so the contexts piled up until late tests found too little free memory for their default sparse-PF budget.
DIFFRACTOM_TEST_MEMLOG=<file>: append this process's GPU memory (nvidia-smi, MiB) after every test."""
import gc
import os
import subprocess

import pytest


@pytest.fixture(autouse=True)
def _release_device_memory(request):
    yield
    try:
        import pyopencl.tools
        pyopencl.tools.clear_first_arg_caches()
    except ImportError:
        pass
    gc.collect()
    log = os.environ.get("DIFFRACTOM_TEST_MEMLOG")
    if log:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            mib = sum(int(l.split(",")[1]) for l in out.splitlines() if l.split(",")[0].strip() == str(os.getpid()))
        except Exception as e:  # noqa: BLE001
            mib = f"? {e}"
        with open(log, "a") as f:
            f.write(f"{request.node.nodeid}\t{mib}\n")
