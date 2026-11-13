#!/bin/bash
# Exp. 16: donor bootstrap, human clocks, donor phases shared across tissues (exp. 14 table),
# covariate effects detached from the rhythmic encoder input (--covariate-encoder-detach).
# Same as exp. 15 otherwise. Launched 2026-11-13 in screen 'chain16'.
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
$P scripts/gtex_bootstrap.py --clock human --phase-table results/gtex/exp14_shared_phase/donor_phases.csv --covariate-encoder-detach --outdir results/gtex/exp16_bootstrap_shared_detach > results/gtex/exp16_bootstrap_shared_detach.driver.out 2>&1
$P scripts/gtex_bootstrap_report.py --boot results/gtex/exp16_bootstrap_shared_detach > results/gtex/exp16_bootstrap_shared_detach/report.out 2>&1
echo "chain done $(date)" >> results/gtex/exp16_chain.log
