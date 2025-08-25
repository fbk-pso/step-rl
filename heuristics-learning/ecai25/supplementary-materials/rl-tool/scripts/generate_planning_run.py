import os
import argparse


SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))


QUEUE = "identical-8"
SLURM_EXTRA_ARGS = "" # example: "-x node87"
NUM_CORES = "4"
TIMEOUT = "600" # seconds
MEMORY_LIMIT = "20000000" # KB
EXECUTABLE = os.path.join(SCRIPTDIR, "planner.sh")

template = '''
[group_test_{}]
num_cores = {}
wctime_limit = {}
rssmem_limit = {}
executable = {}
parameters = {}
basedir = {}
instances_file = {}
tarball = {}
'''

retrive_res = "python3 ~/benchtools/exec/notifier.py --db sqlite:///db --group test_{name} --tarball $1/{name}.tar.bz2\n"


# To customize
def get_symbolic_parameters():
    return [("hff", "solvehsym --heuristic hff --weight 0.8"), ("hadd", "solvehsym --heuristic hadd --weight 0.8")]


# To customize
def get_rl_parameters(domain, config_path, run_path):
    for m in os.listdir(os.path.join(run_path, 'models')):
        model = os.path.join(run_path, 'models', m, 'model.pt')
        config = os.path.join(config_path, 'planning.json')
        yield (m+"_rlh", f"solverlh --model {model} --deltah-bin 1200 --domain {domain} --weight 0.8 --json {config}")
        yield (m+"_rlrank", f"solverlrank --model {model} --domain {domain} --weight 0.8 --heuristic hff --json {config}")


def generate_conf(basedir, cluster):
    """
    param basedir: input directory
    param cluster: a boolean specifying if the experiment is to be run in local or on cluster with slurm
    """

    if cluster:
        env = '''[environment]
db_uri=db
create_db=True
scheduler = slurm
slurm_queue = {}
slurm_resources = {}
dump_scheduler_script={}/dbgscript.txt
slurm_keep_job_output = false
        '''
    else:
        env = '''[environment]
db_uri=db
create_db=True
scheduler = make
dump_scheduler_script={}/dbgscript.txt
        '''

    timeout = str(int(TIMEOUT) + 60)
    memory_limit = str(int(MEMORY_LIMIT) + 1024*1024)

    res_dir = os.path.join(basedir, 'planning_res')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(os.path.join(basedir, "test.conf"), "w") as f0, open(os.path.join(basedir, "retrive_res.sh"), "w") as f1:
        if cluster:
            f0.write(env.format(QUEUE, SLURM_EXTRA_ARGS, basedir))
        else:
            f0.write(env.format(basedir))
        f1.write("#!/bin/sh\n\n")
        f1.write('if [ -z $1 ]; then\n    echo 1>&2 "$0: an argument is needed"\n    exit 2\nfi\n')
        f1.write(f"cd {basedir}\n")
        f1.write(f"mkdir $1\n\n")
        for s in os.listdir(os.path.join(basedir, 'instances_sets')):
            set_path = os.path.abspath(os.path.join(basedir, 'instances_sets', s))
            instances_dir = os.path.join(set_path, "testing_set")
            instances = os.path.join(set_path, "testing_set.txt")
            with open(instances, "w") as f2:
                for i in os.listdir(instances_dir):
                    f2.write(i+'\n')
            for name, parameters in get_symbolic_parameters():
                f0.write(template.format(f"{s}_{name}", NUM_CORES, timeout, memory_limit, EXECUTABLE, parameters, instances_dir, instances, os.path.join(res_dir, f"{s}_{name}.tar.bz2")))
                f1.write(retrive_res.format(name=f"{s}_{name}"))
            for c in os.listdir(os.path.join(set_path, 'learning_configurations')):
                config_path = os.path.join(set_path, 'learning_configurations', c)
                for r in os.listdir(os.path.join(config_path, 'runs')):
                    run_path = os.path.join(set_path, 'learning_configurations', c, 'runs', r)
                    for name, parameters in get_rl_parameters(os.path.join(basedir, "domain.anml"), config_path, run_path):
                        f0.write(template.format(f"{s}_{c}_{r}_{name}", NUM_CORES, timeout, memory_limit, EXECUTABLE, parameters, instances_dir, instances, os.path.join(res_dir, f"{s}_{c}_{r}_{name}.tar.bz2")))
                        f1.write(retrive_res.format(name=f"{s}_{c}_{r}_{name}"))
        f1.write(f"\npython3 {SCRIPTDIR}/analyze_planning.py -i {basedir} -d $1\n\n")
        f1.write(f"rm -r $1\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)
    parser.add_argument('-c', action='store_true')

    args, _ = parser.parse_known_args()

    generate_conf(os.path.abspath(args.i), args.c)


if __name__ == '__main__':
    main()
