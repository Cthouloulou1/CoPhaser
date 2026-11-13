#!/bin/bash
# Step 9: donor bootstrap with the human clocks and donor phases shared across tissues
# (results/gtex/exp14_shared_phase/donor_phases.csv). Launched 2026-11-11 in screen 'chain15'.
cd /workspaces/eyerly/CoPhaser
P=/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python
$P scripts/gtex_bootstrap.py --clock human --phase-table results/gtex/exp14_shared_phase/donor_phases.csv --outdir results/gtex/exp15_bootstrap_shared > results/gtex/exp15_bootstrap_shared.driver.out 2>&1
$P scripts/gtex_bootstrap_report.py --boot results/gtex/exp15_bootstrap_shared > results/gtex/exp15_bootstrap_shared/report.out 2>&1
echo "chain done $(date)" >> results/gtex/exp15_chain.log
