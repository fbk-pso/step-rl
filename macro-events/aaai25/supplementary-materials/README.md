# Supplementary materials for paper *Automatic Selection of Macro-Events for Heuristic-Search Temporal Planning*

The files are organized as follows.

* testing_sets: contains testing instances for each benchmark and for each set (4-fold splitting).

* tamerlite: the planner TAMER (possibly using macro-events).

* planning_results: contains complete (all approaches) planning results for each benchmark and each heuristics. 

* plots: contains all plots, scatter plots for every couple of approaches and cactus plots for every benchmark and every heuristics.

* plots_all.sh: script to generate all plots.

* scripts: contains other python scripts for experiments reproducibility

* optimal_sets_macros: for each benchmark and for each set contains optimal set of macros selected for every usage. For lack of space, it is possible to find all candidate macro-events at the following link : https://zenodo.org/records/13342809 


## Commands

To run a planner on an ANML instance: add tamerlite dir to the PYTHONPATH. Then, use the following command: 
```
python3 scripts/runner.py solvehsym -r <path to output dir> -d testing_sets/<benchmark>/domain.anml -t testing_sets/<benchmark>/<set>/testing_set/<instance.anml> -z <heuristics> 
```

To launch solver with macros add the flags: 
```
-ma optimal_sets_macros/<benchmark>/<set>/best_macros_<macros_usage>.csv -u <macros_usage>
```

where
- benchmark: 'kitting', 'majsp'.
- heuristics: 'hadd', 'hff'.
- macros_usage: 'FA-', 'FA+', 'PA-', 'PA+'.


To run macros selection: 
```
python3 scripts/select_macros.py -i macros_selection/<benchmark>/<set>/macros.csv -j macros_selection/<benchmark>/<set>/episodes.log.csv -o <path to output dir> -d <benchamrk> -u <macros_usage>
``` 

For cases 'PA+' 'FA+', add the flag -t <timeout> (we used timeout=1800).











