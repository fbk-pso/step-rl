import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import re
import argparse
from collections import deque
from os import listdir
from os.path import join, isfile, dirname, abspath, exists


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)
    args, _ = parser.parse_known_args()

    episodes_files = []
    for s in listdir(join(args.i, 'instances_sets')):
        set_path = abspath(join(args.i, 'instances_sets', s))
        for c in listdir(join(set_path, 'learning_configurations')):
            config_path = join(set_path, 'learning_configurations', c)
            for r in listdir(join(config_path, 'runs')):
                run_path = join(config_path, 'runs', r)
                episodes = join(run_path, 'episodes.log.csv')
                episodes_files.append(episodes)

    n_episodes = float('inf')
    episodes_list = []
    for episodes_file in episodes_files:
        episodes = []
        with open(episodes_file) as f:
            lines = f.readlines()[1:]
            for l in lines:
                v = re.split(r'\s*,\s*', l)
                if v[5] == 'True':
                    episodes.append(1)
                else:
                    episodes.append(0)
        n_episodes = min(n_episodes, len(episodes))
        episodes_list.append(episodes)

    plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 32})
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('Solving Rate')
    plt.ylim(0, 1)
    plt.xlim(0, n_episodes)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    y_sum = [0 for x in range(n_episodes)]
    for episodes in episodes_list:
        cumulator = deque(maxlen=1000)
        y = []
        for e in episodes:
            cumulator.append(e)
            y.append(sum(cumulator) / len(cumulator))
        y = [y[i] for i in range(0, n_episodes, 100)] + [y[-1]]
        plt.plot(list(range(1, n_episodes+1, 100)) + [n_episodes], y, color='lightblue')
        y_sum = [sum(x) for x in zip(y_sum, y)]
    y = [x/len(episodes_list) for x in y_sum]
    plt.plot(list(range(1, n_episodes+1, 100)) + [n_episodes], y, color='b')

    # plt.savefig(join(args.i, 'learning.pgf'), dpi=100)
    plt.savefig(join(args.i, 'learning.png'), dpi=100)


if __name__ == '__main__':
    main()
