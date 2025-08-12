import numpy as np
import importlib
import unified_planning as up
import unified_planning.model
from unified_planning.io import ANMLReader
from unified_planning.model.walkers import Simplifier
from unified_planning.model.fluent import get_all_fluent_exp
from tamerlite.encoder import get_encoders, StateEncoder
from tamerlite.core import HAdd, HFF, Effect, Event, simplify
from fractions import Fraction
from rltool.utils import LRUDict


class GeneralSimulator:
    def __init__(self, config):
        self.config = config
        module = importlib.import_module(config.problem_package)
        problem_params = [tuple(p) if type(p)==list else p for p in config.problem_params]
        problem_generator = getattr(module, config.problem_class)(*problem_params)
        domain = problem_generator.domain()
        self.domain = domain
        self.n_problems = problem_generator.size()
        self.problems_iter = problem_generator.get_problems()
        self.environment = domain.environment
        self.all_objects = domain.all_objects
        self.encoder, self.general_state_encoder, self.map_back_ai = get_encoders(domain)
        self.static_fluents = []
        for sf in self.encoder.problem.get_static_fluents():
            self.static_fluents.extend([str(f) for f in get_all_fluent_exp(self.encoder.problem, sf)])

    def get_state_geometry(self):
        return self.general_state_encoder.state_geometry


def get_applicable_actions(general_simulator, initial_values, objects):
    all_objects = general_simulator.all_objects
    map_back_action_instance = general_simulator.map_back_ai
    problem = general_simulator.domain.clone()

    for f, v in initial_values.items():
        problem.set_initial_value(f, v)

    em = problem.environment.expression_manager
    for ut in problem.user_types:
        fname = f"_is_active_{ut.name}"
        f = general_simulator.encoder.problem.fluent(fname)
        problem.add_fluent(f, default_initial_value=False)
        for obj in problem.all_objects:
            if obj.type != ut:
                continue
            if obj in objects:
                problem.set_initial_value(f(obj), em.TRUE())
                initial_values[f(obj)] = em.TRUE()
            else:
                initial_values[f(obj)] = em.FALSE()

    inactive_objects = [obj for obj in all_objects if obj not in objects]

    # Compute the potentially applicable actions
    actions = []
    simplifier = Simplifier(problem.environment, problem)
    for action in general_simulator.encoder.problem.actions:
        ai = map_back_action_instance(action())
        applicable = True
        for obj in ai.actual_parameters:
            if obj.is_object_exp() and obj.object() in inactive_objects:
                applicable = False
                break
        if applicable:
            if isinstance(action, up.model.InstantaneousAction):
                conditions = action.preconditions
            else:
                conditions = action.conditions.values()
            for lc in conditions:
                nc = simplifier.simplify(em.And(lc))
                if nc.is_false():
                    applicable = False
                    break
        if applicable:
            actions.append(action.name)

    return actions


def simplify_static_fluents(action, events, subs):
    new_events = []
    for i, (t, event) in enumerate(events):
        new_c = simplify(event.conditions, subs)
        new_e = []
        for eff in event.effects:
            new_e.append(Effect(eff.fluent, simplify(eff.value, subs)))
        new_event = Event(action, i, new_c, tuple(), tuple(), tuple(new_e))
        new_events.append((t, new_event))
    return new_events


class Simulator:
    def __init__(self, general_simulator, initial_values, objects, goals):
        self._max_step = general_simulator.config.max_step
        self._deltah_cnt = general_simulator.config.deltah_cnt
        self._reward_signal = general_simulator.config.reward_signal
        self._reward_crumb = general_simulator.config.reward_crumb
        encoder = general_simulator.encoder
        full_initial_values = dict(general_simulator.domain.initial_values)
        full_initial_values.update(initial_values)
        self._ss = encoder.search_space
        self._goal = encoder.goals(goals)
        self._actions = get_applicable_actions(general_simulator, full_initial_values, objects)
        self._initial_state = encoder.initial_state(full_initial_values)
        self._state_encoder = StateEncoder(general_simulator.environment, general_simulator.general_state_encoder, full_initial_values, goals)
        subs = {k: v for k, v in self._initial_state.items() if k in general_simulator.static_fluents}
        events = {a: simplify_static_fluents(a, e, subs) for a, e in encoder.events.items() if a in self._actions}
        if general_simulator.config.learning_heuristic == "hadd":
            self._h = HAdd(encoder.fluents, encoder.objects, events, self._goal, False, False)
        elif general_simulator.config.learning_heuristic == "hff":
            self._h = HFF(encoder.fluents, encoder.objects, events, self._goal, False, False)
        self._sat_goals = set([tuple([str(i) for i in s]) for s in self._ss.subgoals_sat(self.get_initial_state(), self._goal)])

    def get_initial_state(self):
        self._ss.reset()
        return self._ss.initial_state(self._initial_state)

    def get_state_as_vector(self, state):
        return np.array(self._state_encoder.get_state_as_vector(state))

    def is_goal_state(self, state) -> bool:
        return self._ss.goal_reached(state, self._goal)

    def get_successor(self, state, action):
        """
        Returns the successor of a state obtained by applying an action.

        :param state: state where to apply action
        :param action: action to apply
        :returns: None if action is not applicable in state, otherwise a tuple of 6 items (next_state, done, reward, next_vstate, h, new_sat_goals) where:
                  - next_state is the next state
                  - terminated is True if next state is a goal
                  - truncated is True if the max number of episodes was reached
                  - with the binary reward signal, reward is +1 if next state is a goal, otherwise 0 plus 0.00001 times the
                    number of satisfied subgoals in next state achieved for the first time in the episode;
                    with the counting reward signal, reward is always -1 plus 0.00001 times the number of satisfied subgoals
                  - next_vstate is the encoded next state
                  - h is the heuristic in next state
                  - new_sat_goals is the set of subgoals that were satisfied at least once from the beginning of the episode in the states visited including next_state
        """
        next_state = self._ss.get_successor_state(state, action)
        if next_state is None:   # action is not applicable in state
            return None
        h = self.get_heuristic_value(next_state)
        goal_reached = self.is_goal_state(next_state)
        if True:
            new_sat_goals = self._sat_goals | set([tuple([str(i) for i in s]) for s in self._ss.subgoals_sat(next_state, self._goal)])
            subgoals_reward = len(new_sat_goals) - len(self._sat_goals)
        else: # a different variant for the subgoals reward
            sat_goals = len(self._ss.subgoals_sat(state, self._goal))
            new_sat_goals = len(self._ss.subgoals_sat(next_state, self._goal))
            subgoals_reward = new_sat_goals-sat_goals
        terminated = goal_reached
        truncated = state.g >= self._max_step
        if self._reward_signal=="bin":
            reward = 1.0 if goal_reached else 0
        else:
            reward = -1.0
        if not goal_reached:
            reward += subgoals_reward*self._reward_crumb
        next_vstate = self.get_state_as_vector(next_state)
        return next_state, terminated, truncated, reward, next_vstate, h, new_sat_goals

    def get_transitions(self, state, h):
        """
        Takes in input a state and its heuristic value.
        Returns a dictionary whose keys are actions applicable in state (N.B. the action which brings from a dead end to the terminal state is represented by an arbitrary action
        among all actions of the problem) and values are corresponding transitions i.e. tuples of 6 items (next_state, done, reward, next_vstate, h, new_sat_goals) where:
        - next_state is None if state is a dead end
        - terminated is True if next state is a terminal state i.e. either next state is a goal, or state is a dead end (has no outgoing transitions)
          or the symbolic heuristic h_sym in state is None
        - truncated is True if the max number of steps for an episode was reached
        - in the old reward schema, reward is -1 if state is a dead end or state has h_sym=None, +1 if next state is a goal,
          otherwise 0 plus 0.00001 times the number of satisfied subgoals in next state achieved for the first time in the episode;
          in the new reward schema, reward is -2delta_h if state is a dead end or state has h_sym=None,
          otherwise -1 plus 0.00001 times the number of satisfied subgoals in next state achieved for the first time in the episode
        - next_vstate is the encoded next state
        - h is the heuristic in next state
        - new_sat_goals is the set of subgoals that were satisfied at least once from the beginning of the episode in the states visited including next_state
        """
        res = {}
        if h is not None:
            for a in self.get_applicable_actions():
                transition = self.get_successor(state, a)
                if transition is not None:
                    res[a] = transition
        if h is None or not res:    # if res is empty i.e. there are no applicable actions i.e. state is a dead end
            dummy_action = self.get_applicable_actions()[0]  # just a placeholder
            if self._reward_signal=="bin":
                dead_end_reward = -1.0
            else:
                dead_end_reward = -2*float(self._deltah_cnt)
            res[dummy_action] = None, True, False, dead_end_reward, None, None, None  # create a dummy transition which brings to the terminal state and gives reward -1
        return res

    def update_sat_goals(self, new_sat_goals):
        self._sat_goals = new_sat_goals

    def get_heuristic_value(self, state):
        return self._h.eval(state, self._ss)

    def reset(self):
        self._sat_goals = set([tuple([str(i) for i in s]) for s in self._ss.subgoals_sat(self.get_initial_state(), self._goal)])

    def get_applicable_actions(self):
        return self._actions


class SimulatorsCache:
    def __init__(self, config, problems):
        self._general_simulator = GeneralSimulator(config)
        self.config = config
        self.problems = problems
        self.simulators = {}

    @property
    def size(self):
        return self._general_simulator.n_problems

    def get_state_geometry(self):
        return self._general_simulator.get_state_geometry()

    def get_simulator(self, problem):
        sim = self.simulators.get(problem, None)
        if sim is None:
            for i in range(len(self.simulators), problem+1):
                sim = self._make_simulator()
                if self.problems and i not in self.problems:
                    sim = None
                self.simulators[i] = sim
        return sim

    def _make_simulator(self):
        initial_values, objects, goals = next(self._general_simulator.problems_iter)
        simulator = Simulator(self._general_simulator, initial_values, objects, goals)
        return simulator
