import matplotlib.pyplot as plt
import argparse
import re
import json
from collections import deque
from os import listdir
from os.path import join, abspath

def do_plot(input, json_file, plot_name, config_id, run_id):
    title = 'config {}, run {}'.format(config_id, run_id)
    MAX_CHARS = 25
    with open(json_file) as f:
        data = json.load(f)
        desc_lines = []
        for line in ('{}={}'.format(k, v) for k, v in data.items()):
            if len(line) > MAX_CHARS:
                line = line[:MAX_CHARS-3] + '...'
            desc_lines.append(line)
        desc = "\n".join(desc_lines)
        output_line = '{}, {}, {}, {}'.format(config_id,
                                              run_id,
                                              data['max_episodes'],
                                              data['max_epsilon'])

    episodes = []
    histogram = {}
    with open(input) as f:
        lines = f.readlines()[1:]
        for l in lines:
            v = re.split(r'\s*,\s*', l)
            instance_id = int(v[1])
            if v[5] == 'True':
                episodes.append(1)
                if instance_id not in histogram:
                    histogram[instance_id] = 0
                histogram[instance_id] += 1
            else:
                episodes.append(0)
        last_line = lines[-1]
        v = re.split(r'\s*,\s*', l)

    tot_episodes = len(episodes)
    tot_solved = 0
    tot_instances_solved = 0
    for k, v in histogram.items():
        if v > 0:
            tot_instances_solved += 1
        tot_solved += v

    output_line += ', {}, {}, {}\n'.format(tot_episodes, tot_solved, tot_instances_solved)

    plt.figure(1, figsize=(40,10))
    plt.clf()
    cumulator = deque(maxlen=1000)
    y = []
    for e in episodes:
        cumulator.append(e)
        y.append(sum(cumulator) / len(cumulator))
    plt.subplot(2, 1, 1)
    plt.xlabel('Episode')
    plt.ylabel('Solving Rates')
    plt.ylim(0, 1)
    plt.tick_params(labelright=True)
    plt.plot(list(range(len(y))), y, color='b')

    plt.subplot(2, 1, 2)
    plt.xlabel('Instance ID')
    plt.ylabel('Solved')
    plt.bar(list(histogram.keys()), histogram.values(), color='r')

    plt.suptitle(title, size=24)
    plt.gcf().text(0.01, 0.5, desc, va='center', ha='left', fontsize=18)

    plt.savefig(plot_name, dpi=100)

    return output_line


def main():
    """
    This function also creates plots and csv files to present the learning results.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)

    args, _ = parser.parse_known_args()

    for s in listdir(join(args.i, 'instances_sets')):
        set_path = abspath(join(args.i, 'instances_sets', s))
        with open(join(set_path, 'res.csv'), 'w') as f:
            f.write('config_id, run_id, max_episodes, max_epsilon, tot_episodes, tot_solved, tot_instances_solved\n')
        for c in listdir(join(set_path, 'learning_configurations')):
            config_path = join(set_path, 'learning_configurations', c)
            learning = join(config_path, 'learning.json')
            for r in listdir(join(config_path, 'runs')):
                run_path = join(config_path, 'runs', r)
                episodes = join(run_path, 'episodes.log.csv')
                plot = join(run_path, 'learning.png')
                res = do_plot(episodes, learning, plot, c, r)
                with open(join(set_path, 'res.csv'), 'a') as f:
                    f.write(res)


if __name__ == '__main__':
    main()
