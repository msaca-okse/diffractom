#!/bin/bash
# Submit a benchmark run to an exclusive A40 (batch job).
#
#   benchmarks/submit.sh [run.py options, e.g. --versions main --suite quick]
#
# Runs on host $BENCH_HOST (default n-62-18-4, where every result so far was measured, so that
# numbers compare); BENCH_AFTER=<job id> starts it after that job has ended.
#
# The benchmark scripts are copied, and the versions resolved to commit hashes, now: the job
# never reads the working copy, so it can be edited or switched to another branch meanwhile.
# Results go to benchmarks/results/ in this repository (commit them) and to $SCRATCH.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH=${BENCH_SCRATCH:-/dtu-compute/msaca/diffractom_bench}
RUN=$SCRATCH/run_$(date +%Y%m%d_%H%M%S)
PY=${BENCH_PYTHON:-/zhome/71/c/146676/miniconda3/envs/textom/bin/python}
mkdir -p "$RUN/harness"
cp "$REPO"/benchmarks/*.py "$RUN/harness/"
HARNESS=$(git -C "$REPO" rev-parse --short=7 HEAD)$(git -C "$REPO" diff --quiet HEAD -- benchmarks || echo "-dirty")

ARGS=("$@")
if [[ ! " $* " =~ " --versions " ]]; then
  ARGS+=(--versions "$("$PY" "$RUN/harness/run.py" --scratch "$RUN" --repo "$REPO" --resolve)")
else  # resolve the given versions now
  for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--versions" ]]; then
      ARGS[$((i+1))]=$("$PY" "$RUN/harness/run.py" --scratch "$RUN" --repo "$REPO" --resolve --versions "${ARGS[$((i+1))]}")
    fi
  done
fi
printf '%q ' "${ARGS[@]}" > "$RUN/args"
AFTER="#"
[ -n "$BENCH_AFTER" ] && AFTER="#BSUB -w \"ended($BENCH_AFTER)\""

bsub <<JOB
#!/bin/bash
#BSUB -J diffractom_bench
#BSUB -q gpua40
#BSUB -m ${BENCH_HOST:-n-62-18-4}
$AFTER
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -M 8000
#BSUB -W 24:00
#BSUB -oo $RUN/job_%J.out
#BSUB -eo $RUN/job_%J.err
export PYOPENCL_CTX=0
cd $RUN
$PY -u $RUN/harness/run.py --scratch $RUN --repo $REPO --results-dir $REPO/benchmarks/results --harness $HARNESS $(cat "$RUN/args")
JOB
echo "run directory: $RUN (log: job_<id>.out)"
