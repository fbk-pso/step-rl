import math
import random
import numpy as np
from collections import deque

from rltool.simulator import SimulatorsCache
from rltool.utils import get_lower_hard
import torch


class Learner:
    def __init__(self, policy, sim_cache, pos_memory, neg_memory, config):
        self._policy = policy
        self._pos_memory = pos_memory
        self._neg_memory = neg_memory
        if config.reward_signal=="cnt":
            self._gamma = 1
        else:
            self._gamma = config.gamma
        self._max_eps = config.max_epsilon
        self._min_eps = config.min_epsilon
        self._batch_size = config.batch_size
        self._deltah_cnt = config.deltah_cnt
        self._reward_signal = config.reward_signal
        self._bootstrap_trunc = config.bootstrap_trunc
        self._residual = config.residual
        self._cumulator = deque(maxlen=100)
        self._episodes = 0
        self._kdecay = math.log(self._min_eps / self._max_eps) / config.max_episodes
        self.eps = self._max_eps
        self.last_reward = None
        self._sim_cache = sim_cache
        n_problems = sim_cache.size
        self.last_episodes = deque(maxlen=max(n_problems, 100))

    def _update_epsilon(self):
        self.eps = self._max_eps * math.exp(self._kdecay * self._episodes)

    def run(self, iid):
        """
        Do an episode of reinforcement learning on a given problem instance.
        The training step is done at the end of the episode.
        The episode terminates when a goal is reached, or a dead end is reached;
        it is truncated when the maximum number of steps has elapsed.

        :param iid: index of the instance to be learnt
        :returns: True if the goal was reached in the episode
        """
        simulator = self._sim_cache.get_simulator(iid)
        state = simulator.get_initial_state()

        vstate = simulator.get_state_as_vector(state)
        h = simulator.get_heuristic_value(state)

        tot_reward = 0
        goal_reached = False
        num_steps = 0
        trace = []
        actions_trace = []

        self._episodes += 1
        self._update_epsilon()

        while True:
            num_steps += 1
            next_steps = simulator.get_transitions(state, h)
            action = self.choose_action(next_steps)
            simulator.update_sat_goals(next_steps[action][-1])

            to_store = []
            actions_trace.append(action)
            for next_state, terminated, truncated, reward, next_pstate, next_h, _ in next_steps.values():
                if terminated:
                    to_store.append((terminated, truncated, reward, None, None))
                elif truncated:
                    to_store.append((terminated, truncated, reward, None, next_h))
                else:
                    to_store.append((terminated, truncated, reward, next_pstate, next_h))
            trace.append((vstate, h, to_store))

            # move the agent to the next state and accumulate the reward
            next_state, terminated, truncated, reward, next_pstate, next_h, _ = next_steps[action]

            state = next_state
            vstate = next_pstate
            h = next_h
            tot_reward += reward

            # if the game is done, break the loop
            if terminated or truncated:
                simulator.reset()
                if state is not None:
                    goal_reached = simulator.is_goal_state(state)
                break
        if goal_reached:
            self._pos_memory.add_trace(trace)
        else:
            self._neg_memory.add_trace(trace)

        self.replay(self._pos_memory)
        self.replay(self._neg_memory)

        self.last_reward = tot_reward
        self.last_num_steps = num_steps
        self._cumulator.append(tot_reward)
        if goal_reached:
            self.last_episodes.append(1)
        else:
            self.last_episodes.append(0)

        return goal_reached, actions_trace


    def replay(self, memory):
        """
        Sample a minibatch from the replay memory and train the model on it.
        If the bootstrap parameter is set to 'const', when the episode is truncated the value of the next state is taken to be zero in the binary
        reward schema and -delta_h in the counting reward schema. If the parameter is set to 'sym', the value is set according to the symbolic heuristic.

        :param memory: an object of type rltool.utils.Memory, a sample from memory is a list of tuples whose first element is an encoded state,
                       second element is the heuristic in the state and third element is a list of tuples (reward, next_vstate) where
                       next_vstate is None if the episode is terminated or truncated
        """
        # arrays/tensors shape annotations:
        # N is the size of batch
        # M is the dimension of the encoded state vector
        # K is the sum of the number of actions (excluded actions for which the next state need not be computed its value) for each state in the batch
        if not memory.empty():
            policy = self._policy
            batch = memory.sample(self._batch_size)   # batch is a python list of length N
            states = np.array([x[0] for x in batch])  # shape (N,M)
            hs = [x[1] for x in batch]
            next_states = []   # shape (K,M)
            for x in batch:
                for terminated, truncated, _, next_vstate, _ in x[2]:
                    if not terminated and not truncated:
                        next_states.append(next_vstate)
            next_states = np.array(next_states)

            if self._residual:
                h_list = []
                for x in batch:
                    for terminated, truncated, _, _, next_h in x[2]:
                        if not terminated and not truncated:
                            h_list.append(next_h)
            else:
                h_list = None
            v_ns = policy.predict_batch(next_states, h_list) # shape (K,1)
            v_ns_iter = iter(v_ns)

            # construct targets y
            y = np.zeros((len(batch), 1))   # shape (N,1)
            for i, (_, _, nexts) in enumerate(batch):
                max_v = None
                for terminated, truncated, reward, _, next_h in nexts:
                    current_v = self.get_action_value(terminated, truncated, reward, v_ns_iter, next_h)
                    if max_v is None or current_v > max_v:
                        max_v = current_v
                y[i] = max_v

            policy.train_batch(states, hs, y) # compute the loss and backpropagate

    def get_action_value(self, terminated, truncated, reward, v_ns_iter, next_h):
        """
        Compute the state-action value function given the reward and the state value function
        :param reward: reward obtained by applying the action
        :param v_ns_iter: an iterable from which to get the value of the next state
        :param next_h: symbolic heuristic in the next state
        :returns: the state-action value
        """
        if terminated:
            v_next_state = 0
        elif not truncated:
            v_next_state = float(next(v_ns_iter)[0])
        elif self._bootstrap_trunc=="const":
            if self._reward_signal=="bin":
                v_next_state = 0
            else:
                v_next_state = -self._deltah_cnt
        elif self._bootstrap_trunc=="sym":
            if self._reward_signal=="bin":
                v_next_state = self._gamma**(next_h-1) if next_h is not None else -1
            else:
                v_next_state = -next_h if next_h is not None else -2*self._deltah_cnt
        value = reward + self._gamma * v_next_state
        return value


    def choose_action(self, res):
        """
        Select action using an epsilon-greedy policy.
        The behaviour policy with probability 1-epsilon selects the greedy action according to the current model
        and with probability epsilon randomly picks an action among the applicable actions with a probability inversely
        proportional to the heuristic value, excluding actions which bring to a state with h=None (if there are only
        such actions then it takes one at random).

        :param res: the dictionary returned by Simulator.get_transitions
        :returns: the chosen action
        """
        # arrays/tensors shape annotations:
        # N is the number of actions leading to nonterminal and nontruncated states
        # M is the dimension of the encoded state vector

        use_heuristic = random.random() < self.eps
        histogram = {}
        to_be_predicted = {}
        h_list = []
        for a, (_, terminated, truncated, reward, next_pstate, h, _) in res.items():
            if use_heuristic:
                if h is not None:
                    histogram[a] = h
            else:
                if not terminated and not truncated:
                    to_be_predicted[a] = next_pstate   # next_pstate is a numpy array of shape (M,)
                    h_list.append(h)

        if use_heuristic:
            if len(histogram) == 0:
                action = random.choice(list(res.keys()))
            elif len(histogram) == 1:
                action = list(histogram.keys())[0]
            else:
                action = get_lower_hard(histogram)
        else:
            if len(to_be_predicted)>0:
                v_ns = self._policy.predict_batch(np.array(list(to_be_predicted.values())), h_list)   # np.array(list(to_be_predicted.values())) has shape (N,M) and v_ns has shape (N,1)
                v_ns_iter = iter(v_ns)
            else:
                v_ns_iter = None
            max_v = None
            argmax_v = None
            for a, (_, terminated, truncated, reward, next_pstate, h, _) in res.items():
                current_v = self.get_action_value(terminated, truncated, reward, v_ns_iter, h)
                if argmax_v is None or current_v > max_v:
                    max_v = current_v
                    argmax_v = a
            action = argmax_v
        return action
