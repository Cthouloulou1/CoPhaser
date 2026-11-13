#!/bin/bash
# Waits for the learned-clock bootstrap (screen boot13), then runs the mouse-clock bootstrap
# and both reports. Launched 2026-11-10 in screen 'chain13'.
cd /workspaces/eyerly/CoPhaser
while ! grep -q "All done" results/gtex/exp13_bootstrap/bootstrap.log 2>/dev/null; do
  screen -ls | grep -q boot13 || break
  sleep 60
done
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_bootstrap_report.py --boot results/gtex/exp13_bootstrap > results/gtex/exp13_bootstrap/report.out 2>&1
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_bootstrap.py --clock mouse --outdir results/gtex/exp13b_bootstrap_mouse > results/gtex/exp13b_bootstrap_mouse.driver.out 2>&1
/home/eyerly/miniconda3/envs/CoPhaser_Env/bin/python scripts/gtex_bootstrap_report.py --boot results/gtex/exp13b_bootstrap_mouse > results/gtex/exp13b_bootstrap_mouse/report.out 2>&1
echo "chain done $(date)" >> results/gtex/exp13_chain.log
