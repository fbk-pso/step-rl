#!/usr/bin/env python3

import argparse
import signal
import sys
import os
import random
import time

from rltool.utils import get_lower_hard, Memory
# from rltool.tfpolicy import Policy
from rltool.ptpolicy import Policy
from rltool.learning import Learner
from rltool.simulator import SimulatorsCache
from rltool.configuration import Config

from typing import Optional


FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def signal_handler(sig, frame):
    sys.exit(0)


def parse_cl():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('command', type=str, choices=['learn', 'plot'])
    parser.add_argument('-f', required=True, type=str,
                        help='The learning.json file with the configurations of the learning phases')
    parser.add_argument('-o', required=True, type=str,
                        help='A folder where to write the resulting models and data')

    args, _ = parser.parse_known_args()
    return args


def get_config(args):
    return Config.learning_args_from_json(args.f)


def log(logfile, cnt, iid, learner, last_solved, elapsed_time, solved,
        unsolved_instances, n_problems, pos_mem, neg_mem, do_print=False):
    logfile.write('{}, {}, {}, {}, {}, {}, {}\n'.format(cnt, iid, learner.last_reward, learner.last_num_steps,
                                                    learner.eps, last_solved,
                                                    elapsed_time))
    logfile.flush()

    if do_print:
        print('Episodes {}'.format(cnt))
        print('Last reward {}'.format(learner.last_reward))
        print('Last solved = {} / {}  -  Tot. solved = {}'.format(sum(learner.last_episodes), len(learner.last_episodes), solved))
        print('Unsolved instances = {} / {}'.format(len(unsolved_instances), n_problems))
        print('Pos_mem_size: {}, Neg_mem_size: {}'.format(pos_mem.size(), neg_mem.size()))
        print('Episodes/s: %.3f' % (cnt / elapsed_time))
        print('-'*50)


def do_plotting(config, log_fname):
    from rltool.plot import Plotter
    sim_cache = SimulatorsCache(config)
    n_problems = sim_cache.size
    Plotter(config, log_fname, n_problems).run()

def execute_learning(config, learner, policy, pos_mem, neg_mem, problems, n_problems, out_dir, logfile, datafile):
    if datafile is not None:
        datafile.write('episode, problem_id, solved, trace\n')

    cnt = 0
    solved = 0
    unsolved_instances = set(problems)
    solved_instances = {i:0 for i in unsolved_instances}
    print('Start learning ({} instances)'.format(n_problems))
    start_time = time.time()
    while cnt < config.max_episodes:
        cnt += 1
        if n_problems == 1:
            iid = 0
        elif solved == 0:
            iid = random.choice(problems)
        else:
            iid = get_lower_hard(solved_instances)

        last_solved = False
        sol, actions_trace = learner.run(iid)

        if sol:
            last_solved = True
            solved += 1
            solved_instances[iid] += 1
            if iid in unsolved_instances:
                unsolved_instances.remove(iid)

        elapsed_time = (time.time() - start_time)

        if datafile is not None:
            trace_str = '|'.join(actions_trace)
            datafile.write('{}, {}, {}, {}\n'.format(cnt, iid, last_solved, trace_str))

        log(logfile, cnt, iid, learner, last_solved, elapsed_time, solved,
            unsolved_instances, n_problems, pos_mem, neg_mem,
            do_print=cnt % 50 == 0)

        if cnt % 25000 == 0:
            network_dir = os.path.join(out_dir, 'models', 'model_{}'.format(cnt))
            os.makedirs(network_dir)
            policy.save(network_dir)

    return cnt

def do_learning(config, log_fname, out_dir):
    if config.problem_filter:
        with open(config.problem_filter) as f:
            problems = [int(l.strip()) for l in f.readlines()]
    else:
        problems = None
    sim_cache = SimulatorsCache(config, problems)
    if problems is None:
        problems = range(sim_cache.size)
    n_problems = len(problems)

    state_geometry = sim_cache.get_state_geometry()
    policy = Policy(state_geometry, config)

    pos_mem = Memory(config.memory_size)
    neg_mem = Memory(config.memory_size)

    learner = Learner(policy, sim_cache, pos_mem, neg_mem, config)

    with open(log_fname, 'wt') as logfile:
        logfile.write('episode, problem_id, reward, num_steps, epsilon, solved, time\n')

        if config.dump_traces:
            with open(os.path.join(out_dir, 'database_traces.csv'), 'wt') as datafile:
                cnt = execute_learning(config, learner, policy, pos_mem, neg_mem, problems, n_problems, out_dir, logfile, datafile)
        else:
            cnt = execute_learning(config, learner, policy, pos_mem, neg_mem, problems, n_problems, out_dir, logfile, datafile=None)

    network_dir = os.path.join(out_dir, 'models', 'model_{}'.format(cnt))
    if not os.path.isdir(network_dir):
        os.makedirs(network_dir)
    policy.save(network_dir)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_cl()
    config = get_config(args)

    log_fname = os.path.join(args.o, 'episodes.log.csv')

    if args.command == 'learn':
        do_learning(config, log_fname, out_dir=args.o)
    elif args.command == 'plot':
        do_plotting(config, log_fname)


if __name__ == '__main__':
    main()
