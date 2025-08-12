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
from types import SimpleNamespace
from rltool.configuration import Config

from unified_planning.shortcuts import *
from unified_planning.engines.results import POSITIVE_OUTCOMES
from unified_planning.io import ANMLReader
from rltool.ptpolicy import Net
from tamerlite.engine import SearchParams, RLParams, MultiqueueParams
from shutil import rmtree

get_environment().factory.add_engine('tamerlite', 'tamerlite.engine', 'TamerLite')
set_credits_stream(None)


SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.environ.get('HOME')
sys.path.append(ROOTDIR)


def jsonize_output(res):
    json_res = {
        "solved": res.status in up.engines.results.POSITIVE_OUTCOMES,
        "plan_size": 0 if res.plan is None else len(res.plan.timed_actions),
        "plan": str(res.plan)
    }
    if res.metrics:
        json_res.update(res.metrics)
    return json_res


def run_tamer_rlh(target, model, domain, deltah_bin, w, other_params):
    r = ANMLReader()
    domain = r.parse_problem(domain)
    problem = r.parse_problem(target)
    other_params.deltah_bin = deltah_bin
    other_params.cache_heuristic_in_state = False
    rl_params = RLParams(domain, model, Net, SimpleNamespace(**vars(other_params)))
    params = SearchParams(search="wastar", heuristic="rl_heuristic", weight=w, rl_params=rl_params)
    with OneshotPlanner(name='tamerlite', params={"search": params}) as planner:
        st = time.time()
        res = planner.solve(problem)
        execution_time = time.time() - st
    return res, execution_time

def run_tamer_rlrank(target, model, domain, w, heuristic, other_params):
    r = ANMLReader()
    domain = r.parse_problem(domain)
    problem = r.parse_problem(target)
    cache_h = other_params.residual and heuristic==other_params.learning_heuristic
    other_params.cache_heuristic_in_state = cache_h
    params_1 = SearchParams(heuristic=heuristic, weight=w, rl_params=None, cache_heuristic_in_state=cache_h)
    rl_params = RLParams(domain, model, Net, other_params)
    params_2 = SearchParams(heuristic="rl_rank", weight=1, rl_params=rl_params)
    mq_params = MultiqueueParams([params_1, params_2])
    with OneshotPlanner(name='tamerlite', params={"search": mq_params}) as planner:
        st = time.time()
        res = planner.solve(problem)
        execution_time = time.time() - st
    return res, execution_time


def run_tamer_hsym(target, w, heuristic):
    r = ANMLReader()
    problem = r.parse_problem(target)
    params = SearchParams(search="wastar", heuristic=heuristic, weight=w, rl_params=None)
    with OneshotPlanner(name='tamerlite', params={"search": params}) as planner:
        st = time.time()
        res = planner.solve(problem)
        execution_time = time.time() - st
    return res, execution_time


def get_outfile_name(target):
    target = '_'.join(target.split('/')[-2:]).replace('.anml', '')
    return target + '.json'


def main():
    """
    Calls the OneshotPlanner solve method with tamerlite engine passing the given inputs and saves the output of planning in a json file.
    General inputs must be passed as command line arguments, while RL inputs can be passed either as command line arguments or by providing
    a json configuration file (the former option overrides the second one). The latter option is used by the script analyze_learning.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', type=str, choices=['solverlh', 'solverlrank', 'solvehsym'], default='solverlh')
    parser.add_argument('-t', '--target', required=True, type=str, help="path to problem anml file")
    parser.add_argument('-r', '--res-dir', type=str, help="directory where to save results file")
    parser.add_argument('-m', '--model', type=str, help="path to neural network state file")
    parser.add_argument('-d', '--domain', type=str, help="path to domain anml file")
    parser.add_argument('-s', '--deltah-bin', type=int, default='1200', help="Maximum depth parameter for the binary reward")
    parser.add_argument('-w', '--weight', type=float, default='0.8', help="weight of A* algorithm")
    parser.add_argument('-z', '--heuristic', type=str, default="hff", help="symbolic heuristic used by solvehsym or by solverlrank in the first queue")   # ignored when cmd=solverlh
    parser.add_argument('-j', '--json', type=str, help="path to json file with configuration for the RL heuristic")

    args, unknown_args = parser.parse_known_args()

    if args.cmd != 'solvehsym':
        learning_opts = Config.planning_args_from_json_or_cmdl(args.json, unknown_args)

    res_dir = args.res_dir
    target = args.target
    model = args.model
    domain = args.domain
    deltah_bin = args.deltah_bin
    w = args.weight
    heuristic = args.heuristic

    if args.cmd == 'solverlh':
        planner_res, execution_time = run_tamer_rlh(target, model, domain, deltah_bin, w, learning_opts)
        res = jsonize_output(planner_res)
        res['overall_time'] = execution_time
        res['model'] = model
        res['target'] = target.split('/')[-1]
        res['w'] = w
        res['deltah_bin'] = deltah_bin
        res.update(learning_opts)
        res['heuristic'] = "rl_heuristic"
    elif args.cmd == 'solverlrank':
        planner_res, execution_time = run_tamer_rlrank(target, model, domain, w, heuristic, learning_opts)
        res = jsonize_output(planner_res)
        res['overall_time'] = execution_time
        res['model'] = model
        res['target'] = target.split('/')[-1]
        res['w'] = w
        res.update(learning_opts)
        res['heuristic'] = "rl_rank + {}".format(heuristic)
    elif args.cmd == 'solvehsym':
        planner_res, execution_time = run_tamer_hsym(target, w, heuristic)
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
