#!/usr/bin/env bash
# Re-run every arm affected by the correctness fixes.
#
# Workers are allocated in proportion to each arm's cost per path: the two
# mixture-utility arms solve a three-component problem at every step and take
# roughly three times as long as the single-regime benchmarks. Sharding is
# deterministic, so no iteration is computed twice and the run can be
# interrupted and resumed.
#
# equalweighted is untouched: it holds 1/N by construction and none of the fixes
# reach it.

set -u

PY="${PYTHON:-$HOME/anaconda3/envs/research/python.exe}"
TRIALS="${TRIALS:-1000}"
LOGDIR="${LOGDIR:-/tmp/benchmark_logs}"

mkdir -p "$LOGDIR"
cd "$(dirname "$0")/../.." || exit 1

launch () {
    local tag=$1 module=$2 shards=$3
    shift 3
    for k in $(seq 0 $((shards - 1))); do
        OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 \
            "$PY" -m "regimeaware.routines.$module" \
            "$@" --trials "$TRIALS" --shard "$k/$shards" \
            > "$LOGDIR/${tag}_${k}.log" 2>&1 &
        echo "launched $tag shard $k/$shards (pid $!)"
    done
}

launch model_estimated model 5 --regimes estimated
launch model_oracle    model 5 --regimes oracle
launch baseline        baseline       2
launch rolling_ols     rolling_ols    2
launch global_min_var  global_min_var 1

echo "waiting for $(jobs -p | wc -l) workers; logs in $LOGDIR"
wait
echo "all arms complete"
