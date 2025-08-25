from collections import deque

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MAX_X = 400
SLEEP_TIME = 0.5


class Plotter:
    def __init__(self, config, log_fname, n_problems):
        self.config = config
        self.log_fname = log_fname
        self.solving_rates = deque(maxlen=MAX_X)
        self.scores = deque(maxlen=MAX_X)
        self.means = deque(maxlen=MAX_X)
        self.cumulator = deque(maxlen=100)

        self.last_episodes_maxlen = max(n_problems, 50)
        self.last_episodes = deque(maxlen=self.last_episodes_maxlen)

        self.solved_instances = {}

        self.x_episodes = deque(maxlen=MAX_X)
        self.counter = 0

        self.epsilons = []

        self.stop = False
        plt.ion()


    def run(self):
        with open(self.log_fname, 'rt') as logfile:
            logfile.readline() # skip header
            while not self.stop:
                s = logfile.read()
                changed = False
                for t in self.parse_data(s):
                    self.update(t)
                    changed = True
                if changed:
                    self.plot()
                plt.pause(SLEEP_TIME)


    def handle_close(self, evt):
        self.stop = True


    def parse_data(self, s):
        for l in s.split('\n'):
            l = l.strip()
            if l:
                d = l.split(', ')
                episode, problem_id = [int(x) for x in d[0:2]]
                reward, epsilon = [float(x) for x in d[2:4]]
                solved = (d[4] == 'True')
                etime = float(d[5])
                yield episode, problem_id, reward, epsilon, solved, etime


    def update(self, data):
        r = data[2]
        self.scores.append(r)
        self.cumulator.append(r)
        self.means.append(sum(self.cumulator) / len(self.cumulator))

        if data[4]:
            assert data[2] >= 1
            o = self.solved_instances.get(data[1], 0)
            self.solved_instances[data[1]] = o + 1

        self.last_episodes.append(1 if data[4] else 0)
        self.solving_rates.append(sum(self.last_episodes) / self.last_episodes_maxlen)

        self.counter += 1
        self.x_episodes.append(self.counter)

        eps = data[3]
        self.epsilons.append(eps)


    def plot(self):
        fig = plt.figure(1, (16, 8))
        fig.canvas.mpl_connect('close_event', self.handle_close)

        plt.clf()

        ax = plt.subplot(2, 2, 1)
        self.make_rewards_plot(ax)

        ax = plt.subplot(2, 2, 2)
        self.make_solved_instances_histogram(ax)

        ax = plt.subplot(2, 2, 3)
        self.make_solving_rates_plot(ax)

        ax = plt.subplot(2, 2, 4)
        self.make_epsion_plot(ax)

        fig.tight_layout()

    def make_rewards_plot(self, ax):
        ax.set_title('Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Duration')
        #ax.set_yscale('log')
        ax.set_ylim((-1, 1))
        ax.plot(self.x_episodes, self.scores, color='g')
        ax.plot(self.x_episodes, self.means, color='b')


    def make_solved_instances_histogram(self, ax):
        ax.set_title('Solved Instances Histogram')
        ax.set_xlabel('Instance ID')
        ax.set_ylabel('Solved')
        ax.bar(list(self.solved_instances.keys()), self.solved_instances.values(), color='r')


    def make_solving_rates_plot(self, ax):
        ax.set_title('%% of solved instances over the last %d' % self.last_episodes_maxlen)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Solving Rates')
        ax.set_ylim((0, 1))
        ax.plot(self.x_episodes, self.solving_rates, color='b')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    def make_epsion_plot(self, ax):
        ax.set_title('Epsilon')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_ylim((0, 1))
        ax.set_xlim((0, self.config.max_episodes))
        ax.plot(range(0, len(self.epsilons)), self.epsilons, color='r')
