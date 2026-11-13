#!/bin/bash
# Human clock (CHIRAL cosinor): exp.12-like sweep, then donor bootstrap, then both reports.
# Launched 2026-11-10 in screen 'chain13h'.
cd /workspaces/eyerly/CoPhaser
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_amplitude_sweep.py --clocks human --fixed-phases --covariate-phase --outdir results/gtex/exp12h_phase_human > results/gtex/exp12h_phase_human/driver.out 2>&1
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_amplitude_report.py --sweep results/gtex/exp12h_phase_human > results/gtex/exp12h_phase_human/report.out 2>&1
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_bootstrap.py --clock human --outdir results/gtex/exp13h_bootstrap_human > results/gtex/exp13h_bootstrap_human.driver.out 2>&1
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_bootstrap_report.py --boot results/gtex/exp13h_bootstrap_human > results/gtex/exp13h_bootstrap_human/report.out 2>&1
echo "chain done $(date)" >> results/gtex/exp13h_chain.log
