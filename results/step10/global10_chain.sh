#!/bin/bash
# Step 10, global test (2026-11-26): waits for the real-phase run of step10b_chain.sh, then
# scripts/gtex_global_rhythms.py on results/step10/gtex/robust_ref12 (8 tissues, 200 bootstrap
# replicates, 200 permutations per covariate family) -> results/step10/gtex/robust_ref12/global/.
# Screen 'global10'; trace in results/step10/global10_chain.log.
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
LOG=results/step10/global10_chain.log
echo "wait $(date)" >> $LOG
until grep -q "real done" results/step10/step10b_chain.log; do sleep 60; done
echo "start $(date)" >> $LOG
OMP_NUM_THREADS=4 $P scripts/gtex_global_rhythms.py --jobs 8 --threads 4 > results/step10/gtex_global.out 2>&1
echo "done rc=$? $(date)" >> $LOG
