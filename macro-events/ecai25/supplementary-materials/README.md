# Supplementary materials for paper *Learning of Lifted Macro-Events for Heuristic-Search Temporal Planning*

Authors: Alessandro La Farciola, Alessandro Valentini, and Andrea Micheli.

The files are organized as follows.

- paper appendix

- unified_planning: unified planning library for python with a slight customization.

- scripts: contains python scripts for experiments reproducibility.

- database_traces: database of recorded traces (from RL).

- database_valid_plans: database of valid plans from which to extract database lifted macros.

- database_candidate_macros: extracted database ground and lifted macros for every domain, every set and every planning approach.

- planning_instances: contains validation and testing instances for each benchmark and for each set (4-fold splitting in learning+validation).

- best_macros: for each domain and each threshold contains the best_macros selected in the 4 different planning approaches to use in testing set.

- planning_results: contains complete planning results for each benchmark and each heuristics (for validation divided in subfolder). 

- cactus_plots: contains cactus plots for every benchmark across every planning approach for the best threshold (best_planner); for every benchmark and every planning approach across thresholds.




## How to run experiments

Install unified_planning. Add current dir and tamerlite to the PYTHONPATH.

- To extract Candidate Ground Macro-Events from database of valid traces:
    ```bash 
    python3 scripts/extract_CME.py -i database_traces/<benchmark>/<set>.csv -o database_candidate_macros/ground/<benchmark>/<set> -m 5 -t 2
    ```
    1. benchmark: 'kitting', 'majsp', 'matchcellar'.
    2. set: 'set_1', 'set_2', 'set_3', 'set_4'.

    It selects the positive traces, remove duplicates, take the shortest '-t' traces for every problem_id and extract (ground) CME. Then it dumps in the directory '-o':
    1. a csv file **database_valid_plans.csv** where from **database_traces.csv** survived only rows used for counting frequencies of CME, with an additional column with length of the trace
    - a csv file **database_CME.csv** with all CME where every row has:
        - macro: string of the type "('light_match_m1'  'mend_fuse_f1')"
        - positive frequency
        - length

- Use the script [**create_database_lifted_macros**](scripts/create_database_lifted_macros.py) to extract Lifted CME from the database of Ground CME. 

    The script take as input:
    1. -i : database of ground CME .csv
    2. -o : output directory
    3. -j : database of valid plans .csv
    4. -u : usage approach ('FA-', 'PA-', 'FA+', 'PA+')
    5. -p : domain.anml

    It extracts all lifted macro-events generated from ground CME, it counts the cumulative utility (wrt to ground macros generated from it), and it dumps in the directory '-o': a json file named **database_lifted_macros_{'usage'}.json** that contains lifted macros, each of whom is a list of dictionary {"action": action's name, "variables": parameters of the lifted action, "cumulative_diff_depth": cumulative utility, "size_ground_ma_in_database": number of ground macros generated in the database}.
    ```bash
    python3 scripts/create_database_lifted_macros.py -i database_candidate_macros/ground/<benchmark>/<set>/database_CME.csv -j database_valid_plans/<benchmark>/<set>/database_valid_plans.csv -o database_candidate_macros/lifted/<benchmark>/<set> -u <macros_usage> -p database_valid_plans/<benchmark>/domain.anml
    ```
    1. benchmark: 'kitting', 'majsp', 'matchcellar'.
    2. set: 'set_1', 'set_2', 'set_3', 'set_4'.
    3. macros_usage: 'FA-', 'FA+', 'PA-', 'PA+'.


- To run empirical selection on used macros: 
    ```bash
    python3 scripts/select_used_macros.py -i planning_results/validation/<benchmark_heuristic>/results.csv -j database_candidate_macros/lifted/<benchmark>/set_1/database_lifted_macros_<macros_usage>.json -u <usage> -o best_macros/<benchmark>/ -p database_valid_plans/<benchmark>/domain.anml -t <threshold>
    ```
    1. benchmark: 'kitting', 'majsp', 'matchcellar'.
    2. heuristic: 'hadd', 'hff'.
    3. macros_usage: 'FA-', 'FA+', 'PA-', 'PA+'.
    4. usage: FA_minus', 'FA_plus', 'PA_minus', 'PA_plus'.
    5. threshold: "0.5,0.5", "0.6,0.6", "0.7,0.7", "0.8,0.8", "0.9,0.9", "1,1".

- To make cactus plot
    ```bash
    python3 scripts/do_cactus_plots.py -i planning_results/complete_testing_results.csv -o cactus_plots/ -x time
    ``` 
