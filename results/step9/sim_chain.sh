#!/bin/bash
# Step 9 simulation: per-tissue vs shared vs oracle phases, 5 tissues, 5 data seeds, jitter 0 / 1 h,
# gtex_train.py with --covariate-encoder-detach. Launched 2026-11-13 in screen 'sim9' after diag16.
cd /workspaces/eyerly/CoPhaser
while [ ! -f /tmp/claude-245447/-workspaces-eyerly/a4a90291-295e-4c47-856b-09c2d8e9db9d/scratchpad/diag16/done.log ]; do sleep 30; done
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/step9_shared_phase_sim.py --outdir results/step9/sim > results/step9/sim.driver.out 2>&1
echo "sim done $(date)" >> results/step9/sim_chain.log
