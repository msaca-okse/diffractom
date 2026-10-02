"""
Benchmark diffractom versions on a matrix of problem sizes: speed and peak GPU / host memory.

    python benchmarks/run.py --scratch /path/to/scratch [--versions v1,v2,...] [--suite default] [--modes gpu,stream]

Each version (any git revision) is exported with `git archive` into <scratch>/trees/<sha>, and
every (version, case, mode) runs in a fresh process with PYTHONPATH=<tree>/src (bench_case.py).
The working copy is never used, so it can be edited, or switched to another branch, while a
run is going. Results are appended, one JSON line per run, to
benchmarks/results/<date>_<gpu>_<host>.jsonl in the repository the runner was started from (and
to <scratch>), so a partial run keeps what was measured. `report.py` turns them into tables.

Submit to the A40 queue with benchmarks/submit.sh.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

# Milestones, newest first (see benchmarks/RESULTS.md for what each one changed).
DEFAULT_VERSIONS = ["main", "fbc8934", "56fe181", "18a7a06", "cce2840", "2b2b9b5", "5a402f0", "bec0dc7"]

BASE = {"K": 10000, "N": 120, "sigma_deg": 0.4, "N_Omega": 360, "N_eta": 360, "rings": 14}
SUITES = {
    # one factor at a time around BASE, each sweep in increasing cost: when a case runs out of
    # memory or time, the rest of its sweep is skipped for that version and mode
    "default": ([{"sweep": "K", "K": k} for k in (1000, 3000, 10000, 30000, 100000)]
                + [{"sweep": "N", "N": n} for n in (64, 200, 400)]
                + [{"sweep": "sigma", "sigma_deg": s} for s in (1.0, 2.0, 4.0)]),
    # larger problems: a 600 x 600 grid, and a sparse PF matrix too large to store (K = 100000,
    # sigma = 1 deg: both CSRs ~37 GB)
    "large": [{"sweep": "large_N", "N": 600}, {"sweep": "large_K", "K": 100000, "sigma_deg": 1.0}],
    # wide basis functions on large grids
    "large_sigma": ([{"sweep": "large_sigma_400", "N": 400, "sigma_deg": s} for s in (2.0, 4.0)]
                    + [{"sweep": "large_sigma_600", "N": 600, "sigma_deg": s} for s in (2.0, 4.0)]),
    "quick": [{"sweep": "quick", "K": 1000, "N": 64}],
}
WARMUP = {"K": 200, "N": 32, "N_Omega": 36, "N_eta": 36}  # compiles and caches the kernels of a version


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout.strip()


def machine():
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).splitlines()
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    return {"host": platform.node(), "gpu": gpu[0] if gpu else "?", "cpu": cpu,
            "cores": len(os.sched_getaffinity(0)), "job": os.environ.get("LSB_JOBID")}


def export_tree(repo, rev, scratch):
    sha = sh(["git", "-C", repo, "rev-parse", "--short=7", rev + "^{commit}"])
    tree = os.path.join(scratch, "trees", sha)
    if not os.path.isdir(os.path.join(tree, "src")):
        os.makedirs(tree, exist_ok=True)
        archive = subprocess.run(["git", "-C", repo, "archive", sha], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tree], input=archive, check=True)
    info = sh(["git", "-C", repo, "log", "-1", "--format=%cI%x09%s", sha]).split("\t", 1)
    return sha, tree, {"rev": rev, "sha": sha, "date": info[0], "subject": info[1]}


def run_case(tree, case, timeout):
    env = {**os.environ, "PYTHONPATH": os.path.join(tree, "src"), "PYTHONNOUSERSITE": "1"}
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "bench_case.py"), json.dumps(case)],
                           capture_output=True, text=True, env=env, timeout=timeout)
        line = next((l for l in p.stdout.splitlines()[::-1] if l.startswith("RESULT ")), None)
        if line:
            res = json.loads(line[7:])
        else:  # killed (e.g. host OOM) or crashed before reporting
            res = {"case": case, "status": "crashed", "returncode": p.returncode, "stderr": p.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        res = {"case": case, "status": "timeout", "timeout_s": timeout}
    res["wall_s"] = time.perf_counter() - t0
    if res.get("diffractom_file") and not res["diffractom_file"].startswith(tree):
        res["status"] = "wrong_tree"  # PYTHONPATH did not win over an installed diffractom
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True, help="directory for exported versions and a copy of the results")
    ap.add_argument("--repo", default=os.path.dirname(HERE), help="git repository to take the versions from")
    ap.add_argument("--versions", default=",".join(DEFAULT_VERSIONS), help="comma-separated git revisions")
    ap.add_argument("--suite", default="default", help=f"comma-separated suites: {', '.join(SUITES)}")
    ap.add_argument("--modes", default="gpu,stream", help="gpu, stream (stream only runs on versions that have it)")
    ap.add_argument("--fista-iters", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300, help="seconds per case (more is recorded as timeout)")
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--harness", default=None, help="commit of the benchmark scripts (recorded in the results)")
    ap.add_argument("--resolve", action="store_true", help="print the versions as commit hashes and exit")
    args = ap.parse_args()
    if args.resolve:
        print(",".join(sh(["git", "-C", args.repo, "rev-parse", "--short=7", v.strip() + "^{commit}"])
                       for v in args.versions.split(",") if v.strip()))
        return

    os.makedirs(args.scratch, exist_ok=True)
    m = machine()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    name = f"{stamp}_{m['gpu'].split(',')[0].replace('NVIDIA ', '').replace(' ', '-')}_{m['host']}.jsonl"
    outs = [os.path.join(args.results_dir, name), os.path.join(args.scratch, name)]
    os.makedirs(os.path.dirname(outs[0]), exist_ok=True)
    harness = args.harness or sh(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"])
    print(json.dumps(m), "->", outs[0], flush=True)

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    cases = [{**BASE, **c} for s in args.suite.split(",") for c in SUITES[s.strip()]]
    modes = [x.strip() for x in args.modes.split(",")]
    for rev in versions:
        sha, tree, vinfo = export_tree(args.repo, rev, args.scratch)
        has_stream = "reserve_coefficient_arrays" in open(
            os.path.join(tree, "src/diffractom/operators/single_phase_forward_operator.py")).read()
        w = run_case(tree, {**BASE, **WARMUP, "mode": "gpu", "fista_iters": 1}, args.timeout)
        print(f"[{datetime.now():%H:%M:%S}] {rev} ({sha}) warm-up: {w['status']}", flush=True)
        for mode in modes:
            if mode == "stream" and not has_stream:
                continue
            stopped = set()  # sweeps that ran out of memory or time
            for case in cases:
                c = {**case, "mode": mode, "fista_iters": args.fista_iters}
                if c["sweep"] in stopped:
                    res = {"case": c, "status": "skipped"}
                else:
                    res = run_case(tree, c, args.timeout)
                    if res["status"] in ("oom", "timeout", "crashed"):
                        stopped.add(c["sweep"])
                res.update(version=vinfo, machine=m, harness=harness, timestamp=datetime.now().isoformat(timespec="seconds"))
                for path in outs:
                    with open(path, "a") as f:
                        f.write(json.dumps(res) + "\n")
                ph = res.get("phases", {})
                it = ph.get("fista", {}).get("per_iteration")
                print(f"[{datetime.now():%H:%M:%S}] {rev:8s} {mode:6s} K={c['K']:<6} N={c['N']:<4} "
                      f"sigma={c['sigma_deg']:<4} {res['status']:8s} pf={res.get('pf_mode')} "
                      f"build={ph.get('build', {}).get('seconds', float('nan')):.1f}s "
                      f"fista/it={it if it is None else round(it, 2)}s gpu={res.get('peak_gpu_mib')}MiB "
                      f"host={round(res.get('peak_host_mib') or 0)}MiB", flush=True)
    shutil.rmtree(os.path.join(args.scratch, "trees"), ignore_errors=True)
    print(f"[{datetime.now():%H:%M:%S}] done", flush=True)


if __name__ == "__main__":
    main()
