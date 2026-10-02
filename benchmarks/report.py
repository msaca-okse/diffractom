"""
The benchmark report: benchmarks/RESULTS.md (tables) and benchmarks/figures/*.png (charts, a light
and a dark version each), from every results file in benchmarks/results/.

    python benchmarks/report.py            write RESULTS.md and the figures
    python benchmarks/report.py --print    print the per-case tables to the terminal instead

When a (version, case, mode) was measured more than once, the newest result counts.
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# What each milestone changed (a version not listed here shows its commit subject).
LABELS = {
    "bec0dc7": "before the speed-ups: dense PF matrix only",
    "5a402f0": "sparse PF matrix (`pf_mode=\"auto\"`)",
    "2b2b9b5": "field-of-view support constraint in FISTA",
    "cce2840": "64-bit indexing in the FISTA, prox and PF kernels",
    "18a7a06": "fused FISTA update: 2 coefficient arrays instead of 4",
    "56fe181": "FISTA coefficients streamed from host memory",
    "fbc8934": "NumPy API, (K, Ny, Nx) layout",
    "2c38f65": "sparse PF matrix generated directly, faster sparse products, store what fits, "
               "Radon in slices, larger batches",
}
SHORT = {"bec0dc7": "first (bec0dc7)", "fbc8934": "previous main (fbc8934)"}

SWEEPS = [  # sweep, varied parameter, label, unit
    ("K", "K", "orientations K", ""),
    ("N", "N", "grid N x N (N translations)", ""),
    ("sigma", "sigma_deg", "basis width sigma", " deg"),
]
BASE = {"K": 10000, "N": 120, "sigma_deg": 0.4}
STATUS = {"oom": "out of memory", "timeout": "> 5 min", "skipped": "skipped", "crashed": "crashed",
          "error": "error", "wrong_tree": "error"}

# validated categorical slots 1-3 (dataviz reference palette), light / dark, and GitHub's surfaces
THEMES = {
    "light": dict(bg="#ffffff", text="#0b0b0b", muted="#52514e", grid="#e6e6e3",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(bg="#0d1117", text="#e6edf3", muted="#9198a1", grid="#262c33",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
MARKERS = ["o", "s", "^"]


def load(files):
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    latest = {}
    for r in rows:
        c = r["case"]
        key = (r["version"]["sha"], c["mode"], c["K"], c["N"], c["sigma_deg"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    return list(latest.values())


def fista(r):
    return r.get("phases", {}).get("fista", {}).get("per_iteration") if r is not None and r["status"] == "ok" else None


def fmt_s(v):
    return f"{v:.3g}" if v < 10 else f"{v:.1f}"


def cell(r, value, fmt=fmt_s):
    if r is None:
        return ""
    if r["status"] != "ok":
        if "INVALID_BUFFER_SIZE" in (r.get("error") or ""):
            return "*array > largest GPU buffer*"
        return f"*{STATUS.get(r['status'], r['status'])}*"
    return "--" if value is None else fmt(value)


def versions_of(rows):
    v = {}
    for r in rows:
        v.setdefault(r["version"]["sha"], r["version"])
    return sorted(v.values(), key=lambda x: x["date"])  # oldest first


def index(rows):
    idx = {}
    for r in rows:
        c = r["case"]
        idx[(r["version"]["sha"], c["mode"], c["K"], c["N"], c["sigma_deg"])] = r
    return idx


def sweep_points(rows, var):
    """Values of var in its sweep, plus the base case (marked)."""
    name = {"K": "K", "N": "N", "sigma_deg": "sigma"}[var]
    vals = {r["case"][var] for r in rows if r["case"].get("sweep") == name}
    vals.add(BASE[var])
    return sorted(vals)


def case_of(var, x):
    return {**BASE, var: x}


def table(rows, idx, versions, mode, var, unit, value, fmt=fmt_s):
    xs = sweep_points(rows, var)
    head = [f"{x:g}{unit}" + (" *" if x == BASE[var] else "") for x in xs]
    best = {}
    for x in xs:
        c = case_of(var, x)
        vals = [value(r) for v in versions if (r := idx.get((v["sha"], mode, c["K"], c["N"], c["sigma_deg"]))) is not None and r["status"] == "ok"]
        vals = [u for u in vals if u is not None]
        best[x] = min(vals) if vals else None
    out = ["| version | " + " | ".join(head) + " |", "|---|" + "---:|" * len(xs)]
    for v in versions[::-1]:
        cells = []
        for x in xs:
            c = case_of(var, x)
            r = idx.get((v["sha"], mode, c["K"], c["N"], c["sigma_deg"]))
            u = value(r) if r is not None and r["status"] == "ok" else None
            s = cell(r, u, fmt)
            cells.append(f"**{s}**" if u is not None and u == best[x] else s)
        runs = [idx.get((v["sha"], mode, *(case_of(var, x)[k] for k in ("K", "N", "sigma_deg")))) for x in xs]
        if any(r is not None and r["status"] not in ("error", "wrong_tree") for r in runs):  # versions without this mode: no row
            out.append(f"| `{v['sha']}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def picture(name, alt):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="figures/{name}_dark.png">\n'
            f'  <img alt="{alt}" src="figures/{name}_light.png">\n</picture>')


# ------------------------------------------------------------------------------------------ figures
def style(ax, th):
    ax.set_facecolor(th["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["grid"])
    ax.tick_params(colors=th["muted"], which="both", labelsize=9)
    ax.grid(True, which="major", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figures(rows, idx, versions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    os.makedirs(FIG, exist_ok=True)
    shas = [v["sha"] for v in versions]
    newest = shas[-1]
    picks = [newest] + [s for s in ("fbc8934", "bec0dc7") if s in shas and s != newest]  # newest: slot 1
    name = lambda s: SHORT.get(s, f"newest ({s})" if s == newest else s)
    num = FuncFormatter(lambda y, _: f"{y:g}")

    for mode_name, th in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": th["text"]})

        # 1. every version on the base case
        fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=150, facecolor=th["bg"])
        style(ax, th)
        allys = []
        for j, (mode, label) in enumerate([("gpu", "arrays on the GPU"), ("stream", "coefficients streamed from host")]):
            pts = [(i, fista(idx.get((s, mode, BASE["K"], BASE["N"], BASE["sigma_deg"])))) for i, s in enumerate(shas)]
            pts = [(i, y) for i, y in pts if y is not None]
            if not pts:
                continue
            allys += [y for _, y in pts]
            ax.plot(*zip(*pts), color=th["series"][j], linewidth=1.6, marker=MARKERS[j], markersize=7,
                    markeredgecolor=th["bg"], markeredgewidth=1.2, label=label, zorder=3)
            ax.annotate(f"{pts[-1][1]:.2f} s", pts[-1], xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8, color=th["text"])
        import math
        ax.set_ylim(10 ** math.floor(math.log10(min(allys))), 10 ** math.ceil(math.log10(max(allys))))
        ax.grid(False, axis="x")
        ax.set_xlim(-0.5, len(shas) - 0.2)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(num)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
        ax.set_xticks(range(len(shas)))
        ax.set_xticklabels([f"{s}\n{v['date'][5:10]}" for s, v in zip(shas, versions)], fontsize=8, color=th["muted"])
        ax.set_ylabel("seconds per FISTA iteration (log)", color=th["muted"], fontsize=9)
        ax.set_title(f"Every version: K = {BASE['K']}, {BASE['N']} x {BASE['N']}, sigma = {BASE['sigma_deg']} deg",
                     loc="left", fontsize=11, color=th["text"])
        leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
        for t in leg.get_texts():
            t.set_color(th["text"])
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"versions_{mode_name}.png"), facecolor=th["bg"])
        plt.close(fig)

        # 2. the sweeps, three versions; 3. GPU memory along K
        for figname, mode, metric, ylab, sweeps in [
                ("sweeps_gpu", "gpu", fista, "seconds per FISTA iteration", SWEEPS),
                ("sweeps_stream", "stream", fista, "seconds per FISTA iteration", SWEEPS),
                ("memory", None, lambda r: r.get("peak_gpu_mib") / 1024 if r["status"] == "ok" and r.get("peak_gpu_mib") else None,
                 "peak GPU memory (GiB)", [("K", "K", "orientations K, arrays on the GPU", ""),
                                           ("K", "K", "orientations K, coefficients streamed", "")])]:
            fig, axes = plt.subplots(1, len(sweeps), figsize=(3.3 * len(sweeps) + 0.6, 3.4), dpi=150,
                                     facecolor=th["bg"], sharey=(figname != "memory"))
            for p, (ax, (sw, var, xlabel, unit)) in enumerate(zip(axes, sweeps)):
                m = mode or ("gpu" if p == 0 else "stream")
                style(ax, th)
                xs = sweep_points(rows, var)
                placed = []  # y of the end labels in this panel
                for i, s in enumerate(picks):
                    pts = []
                    for x in xs:
                        c = case_of(var, x)
                        r = idx.get((s, m, c["K"], c["N"], c["sigma_deg"]))
                        y = metric(r) if r is not None else None
                        if y is not None:
                            pts.append((x, y))
                    if not pts:
                        continue
                    ax.plot(*zip(*pts), color=th["series"][i], linewidth=1.6, marker=MARKERS[i], markersize=6,
                            markeredgecolor=th["bg"], markeredgewidth=1.2, label=name(s), zorder=3)
                    y_end = pts[-1][1]
                    if all(abs(y_end / y - 1) > 0.15 for y in placed):  # no colliding labels
                        placed.append(y_end)
                        ax.annotate(f"{y_end:.2g}", pts[-1], xytext=(5, 0), textcoords="offset points",
                                    va="center", fontsize=8, color=th["text"])
                ax.set_xscale("log")
                if figname != "memory":
                    ax.set_yscale("log")
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
                ax.set_xticks(xs)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.xaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
                ax.yaxis.set_major_formatter(num)
                ax.yaxis.set_minor_formatter(FuncFormatter(lambda *a: ""))
                ax.set_xlabel(xlabel + (" (deg)" if unit else ""), color=th["muted"], fontsize=9)
                ax.margins(x=0.12)
            axes[0].set_ylabel(ylab + (" (log)" if figname != "memory" else ""), color=th["muted"], fontsize=9)
            handles, labels = axes[0].get_legend_handles_labels()
            leg = fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper left", ncol=len(labels),
                             bbox_to_anchor=(0.0, 0.91))
            for t in leg.get_texts():
                t.set_color(th["text"])
            title = {"sweeps_gpu": "One factor at a time, arrays on the GPU",
                     "sweeps_stream": "One factor at a time, coefficients streamed from host memory",
                     "memory": "Peak GPU memory of the process along K"}[figname]
            fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=11, color=th["text"])
            fig.tight_layout(rect=(0, 0, 1, 0.82))
            fig.savefig(os.path.join(FIG, f"{figname}_{mode_name}.png"), facecolor=th["bg"])
            plt.close(fig)


# ------------------------------------------------------------------------------------------ markdown
def summary(rows, idx, versions):
    """The newest version against the one before it, case by case."""
    if len(versions) < 2:
        return []
    new = versions[-1]["sha"]
    old = next((v["sha"] for v in versions[::-1][1:] if any(r["version"]["sha"] == v["sha"] and r["case"]["mode"] == "stream" for r in rows)),
               versions[-2]["sha"])
    out = ["", f"## Newest (`{new}`) against the previous main (`{old}`)", "",
           "Seconds per FISTA iteration; speed-up = previous / newest (below 1: slower).", "",
           "| case | mode | previous | newest | speed-up |", "|---|---|---:|---:|---:|"]
    cases = sorted({(r["case"]["K"], r["case"]["N"], r["case"]["sigma_deg"]) for r in rows if r["version"]["sha"] == new})
    for mode in ("gpu", "stream"):
        for K, N, sig in cases:
            a, b = idx.get((old, mode, K, N, sig)), idx.get((new, mode, K, N, sig))
            if a is None or b is None:
                continue
            fa, fb = fista(a), fista(b)
            sp = f"{fa / fb:.2f}x" if fa and fb else ""
            if fa and fb and fa / fb < 0.95:
                sp = f"**{sp}** (slower)"
            out.append(f"| K = {K}, {N} x {N}, sigma = {sig:g} | {mode} | {cell(a, fa)} | {cell(b, fb)} | {sp} |")
    return out



def markdown(rows, idx, versions, files):
    m = rows[0]["machine"]
    out = ["# Benchmarks",
           "",
           "Speed and peak memory of diffractom's texture-tomography reconstruction across versions, on "
           f"one machine: {m['gpu'].split(',')[0]} (48 GB), {m['cpu']}, host `{m['host']}`. "
           "Generated by `benchmarks/report.py` from `benchmarks/results/*.jsonl`; how to run them: "
           "[benchmarks/run.py](run.py), [benchmarks/submit.sh](submit.sh).",
           "",
           "**The problem.** Synthetic, so only the sizes matter: K random orientations (Gaussian basis "
           "functions of width sigma), an N x N grid scanned with N translations, 360 rotation angles "
           "(0-180 deg), 360 azimuthal bins, 14 aluminium rings. Around the base case "
           f"K = {BASE['K']}, {BASE['N']} x {BASE['N']}, sigma = {BASE['sigma_deg']} deg, one factor is "
           "varied at a time.",
           "",
           "**Measured**, each case in a fresh process: building the operator (including the PF matrix), "
           "one forward and one adjoint projection, one power iteration, and FISTA-Huber (non-negative); "
           "the tables show the time per FISTA iteration. Two modes: *GPU* (coefficients and data are "
           "pyopencl arrays on the GPU) and *streamed* (NumPy arrays; the coefficients stay in host "
           "memory and pass through the GPU batch by batch). Peak memory is that of the process "
           "(GPU: nvidia-smi, includes the ~0.4 GB CUDA context; host: RSS). A case over 5 minutes is "
           "stopped, and the rest of its sweep skipped.",
           "",
           "## Versions",
           "",
           "| version | date | what changed |",
           "|---|---|---|"]
    for v in versions[::-1]:
        out.append(f"| `{v['sha']}` | {v['date'][:10]} | {LABELS.get(v['sha'], v['subject'])} |")
    out += summary(rows, idx, versions)
    out += ["", "## Every version on the base case", "", picture("versions", "Seconds per FISTA iteration of every version on the base case"), ""]

    for mode, title in [("gpu", "Arrays on the GPU"), ("stream", "Coefficients streamed from host memory")]:
        out += [f"## {title}", "", picture(f"sweeps_{mode}", f"Seconds per FISTA iteration along each sweep, {title.lower()}"), "",
                "Seconds per FISTA iteration; the fastest version per column in bold; * the base case.", ""]
        for sw, var, label, unit in SWEEPS:
            fixed = ", ".join(f"{k} = {BASE[k]:g}" for k in ("K", "N", "sigma_deg") if k != var).replace("sigma_deg", "sigma")
            out += [f"**{label}** ({fixed})", "", table(rows, idx, versions, mode, var, unit, fista), ""]

    out += ["## Memory", "", picture("memory", "Peak GPU memory along K"), ""]
    for mode, title in [("gpu", "GPU"), ("stream", "streamed")]:
        for what, key, scale in [("GPU", "peak_gpu_mib", 1 / 1024), ("host", "peak_host_mib", 1 / 1024)]:
            val = lambda r, key=key, scale=scale: (r.get(key) or 0) * scale if r is not None and r["status"] == "ok" else None
            out += [f"**Peak {what} memory (GiB), {title} mode, orientations K** (N = {BASE['N']}, sigma = {BASE['sigma_deg']})", "",
                    table(rows, idx, versions, mode, "K", "", val, fmt=lambda u: f"{u:.1f}"), ""]

    large = sorted({(r["case"]["K"], r["case"]["N"], r["case"]["sigma_deg"]) for r in rows
                    if str(r["case"].get("sweep", "")).startswith("large")})
    if large:
        out += ["## Larger problems", "",
                "Seconds per FISTA iteration (and how the PF matrix was applied: dense; sparse with both CSR "
                "copies stored; only the adjoint copy stored; some or all batches generated in every call).", ""]
        head = [f"K = {K}, {N} x {N}, sigma = {s:g}, {mode}" for (K, N, s) in large for mode in ("gpu", "stream")]
        out += ["| version | " + " | ".join(head) + " |", "|---|" + "---:|" * len(head)]
        for v in versions[::-1]:
            cells = []
            for (K, N, s) in large:
                for mode in ("gpu", "stream"):
                    r = idx.get((v["sha"], mode, K, N, s))
                    c = cell(r, fista(r) if r else None)
                    if r is not None and r["status"] == "ok":
                        how = r.get("pf_mode") + (f", {r['pf_storage']}" if r.get("pf_storage") else "")
                        c += f" ({how})"
                    cells.append(c)
            if any(cells):
                out.append(f"| `{v['sha']}` | " + " | ".join(cells) + " |")
        out.append("")

    out += ["## All results", "", "<details><summary>Every case: build, forward, adjoint, FISTA, memory</summary>", "",
            "| version | mode | K | N | sigma | status | PF | build (s) | forward (s) | adjoint (s) | FISTA / it (s) | GPU (GiB) | host (GiB) |",
            "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    order = {v["sha"]: i for i, v in enumerate(versions)}
    for r in sorted(rows, key=lambda r: (-order[r["version"]["sha"]], r["case"]["mode"], r["case"]["K"], r["case"]["N"], r["case"]["sigma_deg"])):
        c, ph = r["case"], r.get("phases", {})
        g = lambda name: ph.get(name, {}).get("seconds")
        f = lambda v: "" if v is None else fmt_s(v)
        gm, hm = r.get("peak_gpu_mib"), r.get("peak_host_mib")
        out.append(f"| `{r['version']['sha']}` | {c['mode']} | {c['K']} | {c['N']} | {c['sigma_deg']:g} | {r['status']} | "
                   f"{r.get('pf_mode') or ''}{(', ' + r['pf_storage']) if r.get('pf_storage') else ''} | {f(g('build'))} | "
                   f"{f(g('forward'))} | {f(g('adjoint'))} | {f(fista(r))} | {f(gm / 1024 if gm else None)} | "
                   f"{f(hm / 1024 if hm else None)} |")
    out += ["", "</details>", "", "Results files: " + ", ".join(f"`{os.path.basename(x)}`" for x in files), ""]
    return "\n".join(out)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "*.jsonl")))
    rows = load(files)
    if not rows:
        print("no results")
        return
    idx = index(rows)
    versions = versions_of(rows)
    if "--print" in sys.argv:
        print(markdown(rows, idx, versions, files))
        return
    figures(rows, idx, versions)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write(markdown(rows, idx, versions, files))
    print(f"wrote {os.path.join(HERE, 'RESULTS.md')} and {FIG}/*.png from {len(rows)} results in {len(files)} files")


if __name__ == "__main__":
    main()
