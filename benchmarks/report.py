"""
Tables from benchmark results: per case, every version's time per FISTA iteration, build time,
peak GPU and host memory.

    python benchmarks/report.py [results/*.jsonl ...]   (default: the newest file in results/)
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v, unit=""):
    return "--" if v is None else (f"{v:.2f}{unit}" if isinstance(v, float) else f"{v}{unit}")


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "*.jsonl")))[-1:]
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    if not rows:
        print("no results")
        return
    m = rows[0]["machine"]
    print(f"{', '.join(os.path.basename(f) for f in files)}: {m['gpu']} on {m['host']}\n")
    key = lambda c: (c["mode"], c.get("sweep", ""), c["K"], c["N"], c["sigma_deg"])
    for ck in sorted({key(r["case"]) for r in rows}):
        mode, sweep, K, N, sigma = ck
        print(f"### {mode}: K={K}, {N}x{N}, sigma={sigma} deg "
              f"(N_Omega={rows[0]['case']['N_Omega']}, N_eta={rows[0]['case']['N_eta']}, rings={rows[0]['case']['rings']})")
        print("| version | date | status | PF | build (s) | forward (s) | adjoint (s) | FISTA / it (s) | peak GPU (GiB) | peak host (GiB) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted((r for r in rows if key(r["case"]) == ck), key=lambda r: r["version"]["date"]):
            ph = r.get("phases", {})
            g = r.get("peak_gpu_mib")
            h = r.get("peak_host_mib")
            print(f"| {r['version']['sha']} | {r['version']['date'][:10]} | {r['status']} | {r.get('pf_mode') or '--'} | "
                  f"{fmt(ph.get('build', {}).get('seconds'))} | {fmt(ph.get('forward', {}).get('seconds'))} | "
                  f"{fmt(ph.get('adjoint', {}).get('seconds'))} | {fmt(ph.get('fista', {}).get('per_iteration'))} | "
                  f"{fmt(g / 1024 if g else None)} | {fmt(h / 1024 if h else None)} |")
        print()


if __name__ == "__main__":
    main()
