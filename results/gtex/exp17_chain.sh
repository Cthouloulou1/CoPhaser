#!/bin/bash
# Exp. 17: as exp. 16 (donor bootstrap, human clocks, exp. 14 shared donor phases,
# --covariate-encoder-detach), plus --no-lambda (lambda(z) collapsed to 0.5 and biased binary
# amplitude effects), with fast deaths as the Hardy reference instead of ventilator cases (0):
#   17a: reference Hardy 1 (violent and fast deaths, 21-31 donors per tissue)
#   17b: Hardy 1 and 2 merged into one level '1-2' (156-234 donors per tissue), reference '1-2'
# Run one after the other (each uses 10 jobs x 6 threads). Launched 2026-11-26 in screen 'chain17'.
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
COMMON="--clock human --phase-table results/gtex/exp14_shared_phase/donor_phases.csv --covariate-encoder-detach --no-lambda"
for V in a b; do
  if [ "$V" = a ]; then OUT=results/gtex/exp17a_ref_hardy1; EXTRA="--hardy-reference 1"
  else OUT=results/gtex/exp17b_ref_hardy12; EXTRA="--merge-hardy 1 2 --hardy-reference 1-2"; fi
  echo "17$V start $(date)" >> results/gtex/exp17_chain.log
  $P scripts/gtex_bootstrap.py $COMMON $EXTRA --outdir $OUT > $OUT.driver.out 2>&1
  $P scripts/gtex_bootstrap_report.py --boot $OUT > $OUT/report.out 2>&1
  echo "17$V done $(date)" >> results/gtex/exp17_chain.log
done
echo "chain done $(date)" >> results/gtex/exp17_chain.log
