#!/bin/bash
# Step 10 runs (launched 2026-11-26 in two screens, alongside exp. 17):
#   screen 'step10sim': simulation validation of the per-gene rhythm GLM (3 data seeds)
#   screen 'step10gtex': GTEx, 8 tissues, shared donor phases (exp. 14), Hardy reference 1-2,
#                        then the phase-permutation negative control on 3 tissues
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
LOG=results/step10/step10_chain.log
mkdir -p results/step10
case "$1" in
  sim)
    echo "sim start $(date)" >> $LOG
    OMP_NUM_THREADS=8 $P scripts/step10_sim.py --threads 8 --outdir results/step10/sim > results/step10/sim.out 2>&1
    echo "sim done rc=$? $(date)" >> $LOG
    ;;
  gtex)
    echo "gtex start $(date)" >> $LOG
    $P scripts/gtex_gene_rhythms.py --jobs 3 --threads 8 --outdir results/step10/gtex/shared_ref12 \
      > results/step10/gtex_shared_ref12.out 2>&1
    echo "gtex done rc=$? $(date)" >> $LOG
    $P scripts/gtex_gene_rhythms.py --jobs 3 --threads 8 --permute-phases 0 \
      --tissues thyroid lung artery_tibial --outdir results/step10/gtex/shared_ref12_perm0 \
      > results/step10/gtex_shared_ref12_perm0.out 2>&1
    echo "gtex perm done rc=$? $(date)" >> $LOG
    ;;
esac
