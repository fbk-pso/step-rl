#!/bin/bash

rm -rf plots

mkdir -p plots/kitting/hff
mkdir -p plots/kitting/hadd
mkdir -p plots/majsp/hff
mkdir -p plots/majsp/hadd

echo "KITTING HFF"
python3 scripts/do_plots_planning.py -o plots/kitting/hff -x time  -i planning_results/planning_res_kitting_hff.csv 2> /dev/null

echo "KITTING HADD"
python3 scripts/do_plots_planning.py -o plots/kitting/hadd -x time  -i planning_results/planning_res_kitting_hadd.csv 2> /dev/null

echo "MAJSP HFF"
python3 scripts/do_plots_planning.py -o plots/majsp/hff -x time  -i planning_results/planning_res_majsp_hff.csv 2> /dev/null

echo "MAJSP HADD"
python3 scripts/do_plots_planning.py -o plots/majsp/hadd -x time  -i planning_results/planning_res_majsp_hadd.csv 2> /dev/null
