# Supplementary materials for the paper _Exploiting Symbolic Heuristics for the Synthesis of Domain-Specific Temporal Planning Guidance using Reinforcement Learning_, ECAI 2025

Authors: Irene Brugnara, Alessandro Valentini, and Andrea Micheli.

## Folder structure 

This repository contains a subfolder `rl-tool` containing the source code and a subfolder `plots` containing the learning curves. In turn, the `rl-tool` folder contains the following subfolders:

- `examples`: contains generators for our three benchmarks (Kitting, MAJSP and MatchCellar)
- `experiments`: contains configuration files to run the experiments and is populated with experiment artifacts and results once training is completed
- `rltool`: contains the source code for training
- `scripts`: contains scripts to run and analyze learning and planning
- `tamerlite`: our temporal planner and rollout simulator implementation

## How to run experiments

### Requirements

As requirements for running the experiments we request to have the following tools installed:

- Python 3.10+
- the Rust toolchain
- Maturin (`pip install maturin`)
- the unified-planning library available at https://github.com/aiplan4eu/unified-planning

Our package `tamerlite` includes a core module written in Rust (under the `rustamer/` directory), which must be compiled using [Maturin](https://github.com/PyO3/maturin) before installing the package:

```bash
cd tamerlite/rustamer
maturin build --release
```

Install the generated wheel (replace `*.whl` with the actual filename):

```bash
pip install target/wheels/*.whl
cd ..
```

Install the remaining Python code of `tamerlite`:

```bash
pip install .
```

### Training

Suppose to have a directory `test/` inside the `experiments` directory with a `configurations.json` file inside. The following instructions show how to run an experiment with that configuration. The folder `experiments/` contains 6 experiment folders to run all three benchmarks with all techniques presented in the paper.
An explanation of the parameters in `configurations.json` file can be found in the `rltool/configuration.py` file.

First of all, run the following commands in order to add to the `PYTHONPATH` the `rl-tool` path and the `tamerlite` path:
```
cd rl-tool
source env_setup.sh
```


Run the learning part of the experiment in this way and produce learning curves to analyze the results:
```
python3 scripts/generate_learning_run.py -i experiments/test/
bash experiments/test/run_learning.sh
python3 scripts/analyze_learning.py -i experiments/test/
```
The learning part produces as artifacts files `model.pt` for each trained neural network containing the model parameters, and configuration files `planning.json` needed to instantiate the trained neural networks. Such files are nested in the `experiments/test/instances_sets` folders and are taken as input by the planning phase.

### Planning

Run the planning part of the experiments in this way:

```
python3 scripts/runner.py {solverlh|solverlrank} --model experiments/test/instances_sets/set_1/learning_configurations/config_1/runs/run_1/models/model_25000/model.pt --domain experiments/test/domain.anml --weight 0.8 --json experiments/test/instances_sets/set_1/learning_configurations/config_1/planning.json --deltah-bin 600 --target=experiments/test/instances_sets/set_1/testing_set/problem_1.anml
```

The argument `{solverlh|solverlrank}` specifies whether to use the trained neural network to extract a heuristic function (`solverlh`) or as a ranking function in the multi-queue planning approach (`solverlrank`). The `--model` argument is the path to the `model.pt` file and the `--json` argument is the path to the `planning.json` file. The `--deltah-bin` argument is the $\Delta_H$ parameter, used only in the case of the binary reward. The `--target` argument is the path to the planning instance.

Finally, run the fully symbolic planner in this way:

```
python3 scripts/runner.py solvehsym --weight 0.8 --target=experiments/test/instances_sets/set_1/testing_set/problem_1.anml
```
