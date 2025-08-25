import jsonargparse

class Config:

    @staticmethod
    def _add_learning_args(p):
        p.add_argument('--max_episodes', required=True, help="Maximum number of episodes", type=int)
        p.add_argument("--max_epsilon", required=True, help="Maximum value of exploration parameter epsilon", type=float)
        p.add_argument("--min_epsilon", required=True, help="Minimum value of exploration parameter epsilon", type=float)
        p.add_argument("--batch_size", required=True, help="Batch size for neural network training", type=int)
        p.add_argument("--memory_size", required=True, help="Size of positive and negative replay buffers", type=int)
        p.add_argument("--max_step", required=True, help="Number of steps before truncation", type=int)
        p.add_argument("--learning_rate", required=True, help="Learning rate of optimizer", type=float)
        p.add_argument("--reward_crumb", required=True, help="Small reward to give when a subgoal is achieved for the first time in episode", type=float)
        p.add_argument("--bootstrap_trunc", required=True, help="When t=delta_RL, bootstrap V(S_t+1) using the symbolic heuristic if 'sym', or setting it to a constant if 'const'", type=str, choices=["sym", "const"])
        p.add_argument("--dump_traces", required=True, help="Option to dump traces of every episode in a csv file", type=bool)

    @staticmethod
    def _add_problems_args(p):
        p.add_argument("--problem_package", required=True, help="Name of the python package containg the problem class", type=str)
        p.add_argument("--problem_class", required=True, help="Name of the python problem class", type=str)
        p.add_argument("--problem_params", required=True, help="Parameters of the python problem class", type=list)
        p.add_argument("--problem_filter", help="Filename containing the instances to consider", type=str)

    @staticmethod
    def _add_learning_planning_args(p):
        p.add_argument("--gamma", required=True, help="Discount rate of the MDP", type=float)
        p.add_argument("--layer_size", required=True, help="Number of neurons in the second hidden layer", type=int)
        p.add_argument("--reward_signal", required=True, help="Reward function (binary or counting)", type=str, choices=["bin", "cnt"])
        p.add_argument("--deltah_cnt", required=False, help="Maximum depth for the counting reward", type=int)
        p.add_argument("--residual", required=True, help="Learn a residual of the heuristic or the heuristic itself", type=bool)
        p.add_argument("--learning_heuristic", required=True, help="Symbolic heuristic used for guidance in the explorative action and for computing the residual", choices=["hadd", "hff"], type=str)

    @staticmethod
    def learning_args_from_json(fname):
        """ Parse arguments for the learning phase from a json configuration file """
        p = jsonargparse.ArgumentParser()
        Config._add_learning_args(p)
        Config._add_learning_planning_args(p)
        Config._add_problems_args(p)
        return p.parse_path(fname)

    @staticmethod
    def planning_args_from_json_or_cmdl(fname, cmdl):
        """
        Parse arguments for the planning phase from a json configuration file
        and override with command line arguments
        :param fname: path to the json configuration file, can be None
        :param cmdl: command line arguments (a list of strings), can be empty
        """
        config_files = []
        if fname is not None:
            config_files.append(fname)
        p = jsonargparse.ArgumentParser(default_config_files=config_files)
        Config._add_learning_planning_args(p)
        return p.parse_args(cmdl)

