#!/bin/bash
# Step 10, corrected (2026-11-26): robust sandwich (HC3) Wald tests, contamination and
# one-group flags (scripts/gtex_gene_rhythms.py), 8 tissues, shared donor phases (exp. 14),
# Hardy 1-2 reference. Two screens in parallel:
#   'step10real': real phases            -> results/step10/gtex/robust_ref12
#   'step10perm': phases permuted (seed 0), all 8 tissues -> results/step10/gtex/robust_ref12_perm0
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
LOG=results/step10/step10b_chain.log
case "$1" in
  real)
    echo "real start $(date)" >> $LOG
    $P scripts/gtex_gene_rhythms.py --jobs 4 --threads 6 --outdir results/step10/gtex/robust_ref12 \
      > results/step10/gtex_robust_ref12.out 2>&1
    echo "real done rc=$? $(date)" >> $LOG
    ;;
  perm)
    echo "perm start $(date)" >> $LOG
    $P scripts/gtex_gene_rhythms.py --jobs 4 --threads 6 --permute-phases 0 \
      --outdir results/step10/gtex/robust_ref12_perm0 > results/step10/gtex_robust_ref12_perm0.out 2>&1
    echo "perm done rc=$? $(date)" >> $LOG
    ;;
esac
