#!/usr/bin/env bash
# Run the full incremental-path experiment across parallel workers.
#
# Workers are allocated in proportion to each arm's cost: the mixture-utility
# arm solves a three-component problem at every step and takes roughly three
# times as long per path as the collapsed-moment arms. Sharding is deterministic,
# so no iteration is computed twice and the run can be interrupted and resumed.

set -u

PY="${PYTHON:-$HOME/anaconda3/envs/research/python.exe}"
TRIALS="${TRIALS:-1000}"
LOGDIR="${LOGDIR:-/tmp/ablation_logs}"

mkdir -p "$LOGDIR"
cd "$(dirname "$0")/../.." || exit 1

launch () {
    local arm=$1 shards=$2
    for k in $(seq 0 $((shards - 1))); do
        OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 \
            "$PY" -m regimeaware.routines.ablation \
            --arm "$arm" --trials "$TRIALS" --shard "$k/$shards" \
            > "$LOGDIR/${arm}_${k}.log" 2>&1 &
        echo "launched $arm shard $k/$shards (pid $!)"
    done
}

launch ck_uni       2
launch ck_multi     2
launch rwls_mvo     2
launch rwls_mixture 6

echo "waiting for $(jobs -p | wc -l) workers; logs in $LOGDIR"
wait
echo "all arms complete"
