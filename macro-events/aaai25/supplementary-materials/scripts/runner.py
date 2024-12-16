#!/usr/bin/env python3

import json
import os
import argparse
import re
import errno
from subprocess import PIPE, Popen
import time
import sys
import unified_planning as up
#from rltool.configuration import Config

from unified_planning.shortcuts import *
from unified_planning.engines.results import POSITIVE_OUTCOMES
from unified_planning.io import ANMLReader
#from rltool.ptpolicy import Net
from tamerlite.engine import SearchParams, RLParams
from shutil import rmtree

get_environment().factory.add_engine('tamerlite', 'tamerlite.engine', 'TamerLite')
set_credits_stream(None)


SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.environ.get('HOME')
sys.path.append(ROOTDIR)


def jsonize_output(res):
    result = {
        "solved": res.status in up.engines.results.POSITIVE_OUTCOMES,
        "plan_size": 0 if res.plan is None else len(res.plan.timed_actions),
        "plan": str(res.plan)
    }
    if res.metrics:
        for key, value in res.metrics.items():
            result[key] = value
    return result


def run(cmd, cwd=None):
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd)
    return  p.communicate()


def absp(fname):
    return os.path.join(ROOTDIR, fname)

def read_macros(file_path):
    macros_list = []

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)

        # Read the rest of the lines
        for line in file:
            # Split the line by commas and strip any extra whitespace
            parts = line.split(',')
            macro = parts[0].strip()
            macros_list.append(re.findall(r"'(.*?)'", macro))

    return macros_list

def run_tamer(target, model, domain, max_step, w, other_params, macros, macros_usage):
    r = ANMLReader()
    domain = r.parse_problem(domain)
    problem = r.parse_problem(target)
    rl_params = RLParams(domain, model, Net, max_step, other_params)
    if macros:
        macros = read_macros(macros)
    params = SearchParams(search="wastar", heuristic="rl_heuristic", weight=w, rl_params=rl_params, macros=macros, macros_usage=macros_usage)
    with OneshotPlanner(name='tamerlite', params={"search": params}) as planner:
        st = time.time()
        res = planner.solve(problem)
        execution_time = time.time() - st
    return res, execution_time


def run_tamer_hsym(target, w, heuristic, macros, macros_usage):
    r = ANMLReader()
    problem = r.parse_problem(target)
    if macros:
        macros = read_macros(macros)
    params = SearchParams(search="wastar", heuristic=heuristic, weight=w, rl_params=None,  macros=macros, macros_usage=macros_usage)
    with OneshotPlanner(name='tamerlite', params={"search": params}) as planner:
        st = time.time()
        res = planner.solve(problem)
        execution_time = time.time() - st
    return res, execution_time


def get_outfile_name(target):
    target = '_'.join(target.split('/')[-2:]).replace('.anml', '')
    return target + '.json'


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=['solve', 'solvehsym'], default='solve')
    parser.add_argument('-t', '--target', required=True, type=str, help="path to problem anml file")
    parser.add_argument('-r', '--res-dir', type=str, help="directory where to save results file")
    parser.add_argument('-m', '--model', type=str, help="path to neural network state file")
    parser.add_argument('-d', '--domain', type=str, help="path to domain anml file")
    parser.add_argument('-s', '--max-step', type=int, default='1200', help="delta_h parameter")
    parser.add_argument('-w', '--weight', type=float, default='0.8', help="weight of A* algorithm")
    parser.add_argument('-z', '--heuristic', type=str, default="hff", help="symbolic heuristic")
    parser.add_argument('-j', '--json', type=str, help="path to json file with configuration for the RL heuristic")
    parser.add_argument('-ma','--macros', type=str, default = None, help="path to csv file of best macro-actions")
    parser.add_argument('-u', '--macros-usage', type=str, default = None, help="usage of macro-actions")

    args, unknown_args = parser.parse_known_args()

    learning_opts = None
    #if unknown_args:
        #learning_opts = Config.planning_args_from_cmdl(unknown_args)
    #if args.json is not None:
        #learning_opts = Config.planning_args_from_json(args.json)

    res_dir = args.res_dir
    target = args.target
    model = args.model
    domain = args.domain
    max_step = args.max_step
    w = args.weight
    heuristic = args.heuristic
    macros = args.macros
    macros_usage = args.macros_usage

    if args.cmd == 'solve':
        planner_res, execution_time = run_tamer(target, model, domain, max_step, w, learning_opts, macros, macros_usage)
        res = jsonize_output(planner_res)
        res['overall_time'] = execution_time
        res['model'] = model
        res['target'] = target.split('/')[-1]
        res['w'] = w
        res['max_step'] = max_step
        res.update(learning_opts)
        res['heuristic'] = "rl_heuristic"
    elif args.cmd == 'solvehsym':
        planner_res, execution_time = run_tamer_hsym(target, w, heuristic, macros, macros_usage)
        res = jsonize_output(planner_res)
        res['overall_time'] = execution_time
        res['target'] = target.split('/')[-1]
        res['w'] = w
        res['heuristic'] = heuristic
    else:
        raise

    if res_dir:
        outfile_name = get_outfile_name(target)
        try:
            os.makedirs(res_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        outfile = os.path.join(res_dir, outfile_name)
        with open(outfile, 'wt') as f:
            json.dump(res, f, indent=2)
            f.write('\n')
    else:
        for k, v in res.items():
            v = str(v).replace('\n', ' ')
            print(f"{k}: {v}")


if __name__ == '__main__':
    main()
