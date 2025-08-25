import os
import numpy as np
import math

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, state_geometry, config):
        super(Net, self).__init__()

        self._residual = config.residual
        self._reward_signal = config.reward_signal
        self._deltah_cnt = config.deltah_cnt
        if hasattr(config, "deltah_bin"):
            self._deltah_bin = config.deltah_bin
        self._gamma = config.gamma

        self.state_geometry = state_geometry
        self.new_state_geometry = []
        self.final_activation = self.get_final_activation()

        self.inputs = []
        c = 0
        first_layer_size_per_input = config.layer_size
        second_layer_size = config.layer_size

        if state_geometry.num_fluents > 0:
            self.dense_begin_fluents = torch.nn.Linear(state_geometry.num_fluents, first_layer_size_per_input)
            self.inputs.append(self.dense_begin_fluents)
            self.new_state_geometry.append(state_geometry.num_fluents)
            c += 1

        if state_geometry.num_actions > 0:
            self.dense_begin_actions = torch.nn.Linear(state_geometry.num_actions, first_layer_size_per_input)
            self.inputs.append(self.dense_begin_actions)
            self.new_state_geometry.append(state_geometry.num_actions)
            c += 1

        if state_geometry.num_constants > 0:
            self.dense_begin_constants = torch.nn.Linear(state_geometry.num_constants, first_layer_size_per_input)
            self.inputs.append(self.dense_begin_constants)
            self.new_state_geometry.append(state_geometry.num_constants)
            c += 1

        if state_geometry.num_goals > 0:
            self.dense_begin_goals = torch.nn.Linear(state_geometry.num_goals, first_layer_size_per_input)
            self.inputs.append(self.dense_begin_goals)
            self.new_state_geometry.append(state_geometry.num_goals)
            c += 1

        if state_geometry.tn_size > 0:
            self.dense_begin_tn = torch.nn.Linear(state_geometry.tn_size, first_layer_size_per_input)
            self.inputs.append(self.dense_begin_tn)
            self.new_state_geometry.append(state_geometry.tn_size)
            c += 1

        self.dense_queue = torch.nn.Linear(first_layer_size_per_input*c, second_layer_size)
        self.dense_final = torch.nn.Linear(second_layer_size, 1)

    def get_final_activation(self):
        if self._reward_signal=="bin":
            if self._residual:
                activation = lambda x: F.softsign(x)*3/2 - 0.5     # output is between -2 and +1
            else:
                activation = F.softsign     # output is between -1 and +1
        else:
            if self._residual:
                activation = lambda x: ((F.softsign(x)*2)-1)*self._deltah_cnt   # output is between -3*delta_h and +delta_h
            else:
                activation = lambda x: (F.softsign(x)-1)*self._deltah_cnt/2*3   # output is between -3*delta_h and 0
        return activation

    def compute_parameterized_part(self, x):
        xs = list(torch.split(x, self.new_state_geometry, dim=1))
        for i, l in enumerate(self.inputs):
            xs[i] = F.relu(l(xs[i]))
        x = torch.cat(xs, dim=1)
        x = F.relu(self.dense_queue(x))
        x = self.dense_final(x)
        x = self.final_activation(x)
        return x

    def forward(self, x, h_list = None):
        """
        Compute the value function by doing the neural network forward and combining with the symbolic part in case of residual learning.

        :param x: torch tensor of encoded states
        :param h: python list of symbolic heuristic computed in the states, same dimension as x
        """
        nn_output = self.compute_parameterized_part(x)
        if self._residual:
            if self._reward_signal == "cnt":
                phi_list = [-h if h is not None else -2*self._deltah_cnt for h in h_list]
            else:
                phi_list = [self._gamma**(h-1) if h is not None else -1 for h in h_list]
            phi_list = torch.tensor(phi_list).unsqueeze(1)
            v = phi_list + nn_output
        else:
            v = nn_output
        return v

    def get_heuristic(self, x, h_list = None):
        """
        Same signature as forward() but returns a heuristic value instead of the value function.
        """
        vs = self(x, h_list).detach()  # calls forward
        res = []
        for v in vs:
            v = float(v[0])
            if self._reward_signal=="bin":
                if v == 0:
                    res.append(float(self._deltah_bin))
                elif v < 0:
                    res.append(float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -v), self._gamma)))))
                else:
                    res.append(float(min(self._deltah_bin, (math.log(min(1, v), self._gamma)+1))))
            else:
                res.append(max(0.000001,-v))
        return res

    def get_rank(self, x, h_list = None):
        """
        Same signature as forward() but returns a rank instead of the value function.
        """
        vs = self(x, h_list).detach()   # calls forward
        res = []
        for v in vs:
            v = float(v[0])
            if self._residual and self._reward_signal=="cnt":
                v -= 3*self._deltah_cnt
            res.append(-v+3.0)
        return res


class Policy:
    def __init__(self, state_geometry, config):
        self._state_geometry = state_geometry
        self._model = Net(state_geometry, config)
        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        #self._train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._train_device = torch.device("cpu")

    def predict_one(self, single_input):
        return self.predict_batch(np.array([single_input]))[0]

    def predict_batch(self, input_batch, h_list):
        """
        Predict the value function for a batch of states using the model.

        :param input_batch: numpy array of the encoded states
        :param h_list: python list of the symbolic heuristic computed in the states of input_batch
        :returns: torch tensor of the value function computed in the states
        """
        with torch.no_grad():
            return self._model(torch.from_numpy(input_batch).float(), h_list).detach()

    def train_batch(self, input_batch, input_h, output_batch):
        """
        Train the value function network on a batch of examples by doing forward and backward pass.

        :param input_batch: numpy array of the encoded current states
        :param input_h: python list of the symbolic heuristic computed in current states
        :param output_batch: numpy array with the targets for the value function update
        """
        # zero the parameter gradients
        self._optimizer.zero_grad()

        # forward + backward + optimize
        pred_output = self._model(torch.from_numpy(input_batch).float(), input_h).to(self._train_device)
        loss = self._criterion(pred_output, torch.from_numpy(output_batch).float().to(self._train_device))
        loss.backward()
        self._optimizer.step()

    def save(self, path):
        torch.save(self._model.state_dict(), os.path.join(path, "model.pt"))
