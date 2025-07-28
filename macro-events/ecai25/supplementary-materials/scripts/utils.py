import random
import time
import re

import json

from collections import deque, OrderedDict
import itertools

from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional

from unified_planning.shortcuts import *
from unified_planning.model.problem import Problem
from unified_planning.model.action import Action, DurativeAction
from unified_planning.model.parameter import Parameter
from unified_planning.plans.plan import ActionInstance
from unified_planning.model.object import Object
from unified_planning.io import ANMLReader
from unified_planning.engines.compilers.grounder import GrounderHelper
from unified_planning.model.types import domain_size, domain_item
# from unified_planning.engines.compilers.utils import get_fresh_name

from simanneal import Annealer
import random
from decimal import Decimal, getcontext
from fractions import Fraction
import pdb
import copy

def expanded_states(l, n):
    return Fraction((n** (l + 1) - 1) / (n - 1))

@dataclass(frozen=True)
class MacroAction:
    actions: Tuple[str]

    def __post_init__(self):
        assert isinstance(self.actions, tuple) > 0, f"{type(self.actions)}"
        assert len(self.actions) > 0

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        if len(self) == 1:
            return self.actions[0]
        else:
            return str(self.actions)

    def __iter__(self):
        for x in self.actions:
            yield x

    def __lt__(self, other):
         return str(self) < str(other)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MacroAction(tuple([self.actions[ii] for ii in range(*key.indices(len(self)))]))
        elif isinstance(key, int):
            if key < 0: # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return MacroAction((self.actions[key],  )) # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def update_sat_goals(self, simulator, next_steps):
        for _, _, _, _, _, next_sat_goals in next_steps[self]:
            simulator.update_sat_goals(next_sat_goals)

    def get_tot_reward(self, next_steps):
        sum_reward = 0
        for _, _, _, reward, _, _ in next_steps[self]:
            sum_reward += reward
        return sum_reward

    def get_last_transition(self, next_steps):
        return next_steps[self][-1]
    
    def increase_count(self, variable, value):
        if len(self) > 1:
            variable = variable + value
        return variable

class MacroActionsMemory:

    def __init__(self, max_memory, max_length, l1, macros_usage, insert_new_macros):
        self._max_memory = max_memory
        self.macro_actions = {}
        self.best_macros = {}
        self.min_threshold = 0
        self._max_length_ma = max_length
        self._l1 = l1
        self.macros_usage=macros_usage
        self._insert_new_macros=insert_new_macros

    def update_best_macros(self, to_update):

        for ma in to_update:
            f_value = self.function_combination(len(ma), self.macro_actions[ma][0], self.macro_actions[ma][1])
            if f_value > self.min_threshold:
                self.best_macros[ma] = f_value
                if len(self.best_macros) > self._max_memory:
                    self.min_threshold = min(self.best_macros.values())
                    index = list(self.best_macros.values()).index(self.min_threshold)
                    self.best_macros.pop(list(self.best_macros.keys())[index], None)
        self.min_threshold = min(self.best_macros.values()) # to be sure that self.min_threshold is right at the end of updates

    def find_frequent_macros(self, input_string: tuple, is_true: bool):

        for i in range(1, self._max_length_ma):
            to_update = []

            window_start = 0
            window_end = i+1

            while window_end <= len(input_string):
                current_substring = input_string[window_start:window_end]
                macro = MacroAction(current_substring)
                to_update.append(macro)
                if macro in self.macro_actions.keys():
                    if is_true:
                        self.macro_actions[macro][0] += 1   #self.macro_actions[macro] is a list [positive_frequency, negative_frequency]
                    else:
                        self.macro_actions[macro][1] += 1
                else:
                    if is_true:
                        self.macro_actions[macro] = [1, 0]
                    else:
                        self.macro_actions[macro] = [0, 1]
                window_start += 1
                window_end += 1
            if self._insert_new_macros != "None":
                self.update_best_macros(to_update)

    def max_diff(self):
        
        max = float('-inf')
        #max_key
        
        # Iterate through the dictionary items
        for key, value in self.macro_actions.items():
            diff = value[0] - value[1]  # Calculate the difference
            if diff > max:
                max = diff
                #max_key = key

        return max

    def function_combination(self, x, y, z):
        diff = y - z
        max = self.max_diff()
        return self._l1 * (diff * Fraction(self._max_length_ma/ max))  + (1 - self._l1) * x

    def sort_macros(self):
        self.macro_actions = dict(sorted(self.macro_actions.items(), key=lambda item: self.function_combination(len(item[0]), item[1][0], item[1][1]), reverse=True))


    def sort_best_macros(self):
        self.best_macros = dict(sorted(self.best_macros.items(), key=lambda item: item[1], reverse=True))

    def print_best_macros_vertically(self):
        for key, value in self.best_macros.items():
            print(f"{key}  : {value}")


def extract_suffix_as_list(prefix, target):
    """
    Extracts the part of `target` string after `prefix` and returns it as a list of substrings.

    Args:
        prefix (str): The prefix string to match.
        target (str): The target string to process.

    Returns:
        list: A list of substrings after the prefix.

    Raises:
        ValueError: If the `target` does not start with the `prefix`.
    """
    prefix = prefix+'_'
    if target.startswith(prefix):
        return target[len(prefix):].split('_')
    else:
        raise ValueError(f"Cannot extract list of objects of ground action {target}")

class GroundMacroEvent:
    """
    Represents a ground macro-event. For now, we represent ground macros with strings (and not with up Action as LiftedMacroEvent) -> to decide if change this 
    """

    def __init__(
        self,
        actions: Union[str, List[str]],
        id : int
    ):
        self.utility : Fraction = 0
        self.diff_depth : Fraction = 0
        self.positive_frequency = 0
        self.validation_frequency = 0
        self.validation_support = set() # set of instances where macro is used 
        self.length = 0
        self._factory = None
        self._id = id

        if isinstance(actions, str):# extract list of actions in the macro
            macro_string = actions.strip('()')
            complete_macros_actions = macro_string.split('  ')
            for i, string in enumerate(complete_macros_actions):
                complete_macros_actions[i] = string.strip("''")
            self.actions = complete_macros_actions
        else:
            self.actions = actions

    def __len__(self):
        return self.length
    
    def __hash__(self):
        # val = 0
        # for i, act in enumerate(self.actions):
        #     val += hash(act) * (i+1)  #order of actions makes macros different
        # return val
        return self._id
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, GroundMacroEvent):
            return False
        if self.actions != other.actions:
            return False
        else:
            return True
        
    def __repr__(self) -> str:
        expr = '('
        for act in self.actions:
            expr += act +', ' 
        expr = expr[:-2] + ')'
        return expr
    
    def set_factory(self, factory):
        self._factory = factory
    
    def get_sub_macros(self):
        for k in range(2, len(self)): # don't include the macro itself
            yield self._factory.make_macro(self.actions[:k])
    
class GroundMacroEventFactory:

    def __init__(self):
        self._macros_memory = {}
        self._count = 0

    def convert_actions_variables(self, actions: Union[str, List[str]]) -> Tuple[Tuple[str, Tuple[str]]]:
        if isinstance(actions, str):# extract list of actions in the macro
            macro_string = actions.strip('()')
            complete_macros_actions = macro_string.split('  ')
            for i, string in enumerate(complete_macros_actions):
                complete_macros_actions[i] = string.strip("''")
            return tuple(complete_macros_actions)
        else:
            return tuple(actions)
        
    def make_macro(self, actions: Union[str, List[str]]) -> GroundMacroEvent:
        actions_converted = self.convert_actions_variables(actions)
        if actions_converted not in self._macros_memory:
            ground_macro = GroundMacroEvent(actions, self._count)
            self._count += 1
            ground_macro.set_factory(self)
            self._macros_memory[actions_converted] = ground_macro
        return self._macros_memory[actions_converted]  

    @property
    def macros_memory(self):
        return self._macros_memory
    

class LiftedMacroEvent:
    """
    Represents a lifted macro-event 
    """

    def __init__(
        self,
        #actions_variables : List[Tuple[str, List[str]]], #for example [('move', ['v1', 'v2']), ('pick', ['v2']) ]
        actions_variables : List[Dict[str, Union[List[Parameter], Action]]], #for example [{'action': move, 'variables' : [v1, v2]), {'action': pick, variables : [v2]} ]
        id : int
        #problem : str #path to problem anml file
    ):
        self._factory = None
        self._id = id
        self.actions_variables = actions_variables
        self.actions = [] #list of Actions
        self.variables = set() #set of Parameters 
        self.cumulative_diff_depth_negative = 0
        self.cumulative_diff_depth_positive = 0
        self.size_ground_ma_in_database = 0
        self.size_ground_exact = 0
        self.validation_frequency = 0 # frequency of the macro use in validation
        self.validation_support = set() #  set of instances with at least a grounding used to find goal

        self.list_of_variables=[]
        for d in self.actions_variables:
            self.actions.append(d['action'])
            for v in d['variables']:
                self.list_of_variables.append(v)
                self.variables.add(v)
        
        self.equivalence_variables_index = set()
        for var in self.list_of_variables:
            self.equivalence_variables_index.add(tuple([i for i, x in enumerate(self.list_of_variables) if x == var]))

    def __repr__(self) -> str:
        
        expr = '['
        for d in self.actions_variables:
            expr += str(d['action'].name)+'('
            for v in d['variables']:
                expr += str(v)+', '
            expr = expr[:-2]+'), '
        expr = expr[:-2]+']'

        return expr

    @property
    def arity(self) -> int:
        return len(self.variables)

    def __hash__(self):
        return self._id

    def set_factory(self, factory):
        self._factory = factory

    @property
    def factory(self):
        return self._factory

    def __eq__(self, other) -> bool:
        if not isinstance(other, LiftedMacroEvent):
            return False
        if [a.name for a in self.actions] != [b.name for b in other.actions]:
            return False
        else:
            #print(f'{self.equivalence_variables_index} vs {other.equivalence_variables_index}   : {self.equivalence_variables_index == other.equivalence_variables_index}')
            return  self.equivalence_variables_index == other.equivalence_variables_index 

    def __len__(self):
        return len(self.actions)
    
    # def __iter__(self):
    #     return self
    
    def to_json(self, planning : Optional[bool] = False):
        res = []
        for d in self.actions_variables:
            clone_dict = {}
            clone_dict['action'] = d['action'].name 
            # clone_dict['variables'] = {v.name : str(v.type) for v in d['variables']}
            clone_dict['variables'] = [v.name for v in d['variables']]
            res.append(clone_dict)

        if not planning:    
            res.append( {'cumulative_diff_depth_negative' : str(self.cumulative_diff_depth_negative)} )
            res.append( {'cumulative_diff_depth_positive' : str(self.cumulative_diff_depth_positive)} )
            res.append( {'size_ground_ma_in_database' : str(self.size_ground_ma_in_database)} )

        return res
    
    def es(self, n, l, grounder_type : str, macros_usage : str) -> Fraction:
        if '-' in macros_usage:
            cumulative_diff_depth = self.cumulative_diff_depth_negative
        else:
            cumulative_diff_depth = self.cumulative_diff_depth_positive
        min_depth = Fraction(l/len(self))
        depth = max(l-cumulative_diff_depth, min_depth)
        assert depth > 0
        if grounder_type == "observed":
            bf = self.size_ground_ma_in_database
        elif grounder_type == 'exact':
            bf = self.size_ground_exact
            if '+' in macros_usage:
                for sub_macro in self.get_sub_macros():
                    bf += sub_macro.size_ground_exact

        else:
            raise NotImplementedError
        return expanded_states(depth, n+bf)
    
    def get_sub_macros(self):
        for k in range(2, len(self)): # don't include the macro itself
            yield self._factory.make_macro(self.actions_variables[:k])
    
    def get_last_sub_macro(self):
        return self._factory.make_macro(self.actions_variables[:-1])
    
    def check_grounding(self, ground_macro: GroundMacroEvent, family: Optional[Tuple[str]] = None) -> bool:

        if not family:
            raise NotImplementedError #to implement: extract family (see create_database...)
        else:
            if not family == tuple([a.name for a in self.actions]) : 
                return False
            else:
                tmp = {}
                for i, d in enumerate(self.actions_variables):
                    if not d['action'].name == family[i]:
                        return False
                    # print(self, ground_macro, extract_suffix_as_list(family[i], ground_macro.actions[i]), d['variables'])
                    # for obj, par in zip(extract_suffix_as_list(family[i], ground_macro.actions[i]), d['variables'], strict=True):
                    for obj, par in zip(extract_suffix_as_list(family[i], ground_macro.actions[i]), d['variables']):
                        objec = tmp.get(par, None)
                        if objec:
                            if not obj == objec:
                                return False
                        else:
                            tmp[par] = obj
        return True

    def is_in_grounded_problem(self, grounder : GrounderHelper):
        for a in self.actions:
            if a.name not in grounder._action_to_params:
                return False
        return True
    
class LiftedMacroEventFactory:

    def __init__(self):
        self._macros_memory = {}
        self._count = 0 #assign progressive id to created LiftedMacroEvent

    def convert_actions_variables(self, actions_variables : List[Dict[str, Union[List[Parameter], Action]]]) -> Tuple[Tuple[str, Tuple[str]]]:
        #return tuple((d['action'].name, tuple(d['variables'])) for d in actions_variables)
        actions = []
        list_of_variables=[]
        for d in actions_variables:
            actions.append(d['action'].name)
            for v in d['variables']:
                list_of_variables.append(v)
        equivalence_variables_index = set()
        for var in list_of_variables:
            equivalence_variables_index.add(tuple([i for i, x in enumerate(list_of_variables) if x == var]))
        return (tuple(actions), frozenset(equivalence_variables_index))

    def make_macro(self, actions_variables : List[Dict[str, Union[List[Parameter], Action]]]) -> LiftedMacroEvent:
        actions_variables_converted = self.convert_actions_variables(actions_variables)
        if actions_variables_converted not in self._macros_memory:
            lifted_macro = LiftedMacroEvent(actions_variables, self._count)
            self._count += 1
            lifted_macro.set_factory(self)
            self._macros_memory[actions_variables_converted] = lifted_macro
        return self._macros_memory[actions_variables_converted]  

    @property
    def macros_memory(self):
        return self._macros_memory

def generate_ground_macros(lifted_macro: LiftedMacroEvent, problem, macros_usage, grounder: Optional[GrounderHelper] = None, prune_actions: Optional[bool] = True):
    """
    Generate the set of ground actions of problem obtained from lifted_macro. 
    problem : up (lifted) problem 
    """

    length_macro = len(lifted_macro)
    parameter_list = list(lifted_macro.variables)
    parameter_indexes = {p:i for i,p in enumerate(parameter_list)}

    # print(parameter_indexes)

    if grounder is None:
        grounder = GrounderHelper(problem, prune_actions=prune_actions)

    #grounder = GrounderHelper(problem, prune_actions=prune_actions)

    all_parameters = get_all_parameters(problem, parameter_list)
    
    for par in all_parameters:
        #assert len(par) == len(parameter_list)
        ground_actions = []
        l=0
        for d in lifted_macro.actions_variables:
            act = d['action']
            var = d['variables']
            params = []
            for v in var:
                ind = parameter_indexes[v]
                params.append(par[ind])
            #assert len(params) == len(var)
            ground_act = grounder.ground_action(act,tuple(params))
            if ground_act is not None:
                l += 1
                ground_actions.append(ground_act.name)
            else:
                break
        if l == length_macro or (l >= 2 and 'PA' in macros_usage):
            yield tuple(ground_actions)
        # if l==4:
        #     print(par)
        #     print(ground_actions)

def get_ground_set(lifted_macro, problem, grounder : Optional[GrounderHelper] = None, prune_actions : Optional[bool] = True):

    return [ma for ma in generate_ground_macros(lifted_macro, problem, grounder, prune_actions)]

class GroundCounter:

    def __init__(self, problem : Problem):
        self.problem = problem

    def get_ground_size(self):
        raise NotImplementedError
    
    def get_ground_problem(self, grounder):

        if not grounder:
            self.grounder = GrounderHelper(self.problem, prune_actions=True)
        else:
            self.grounder = grounder
        self.ground_actions = set()

        for old_action in self.problem.actions:
            for grounded_params in self.grounder.get_possible_parameters(old_action):
                new_action = self.grounder.ground_action(old_action, grounded_params)
                if new_action is not None:
                    self.ground_actions.add(new_action.name)

class GroundCounterProblem(GroundCounter):

    def __init__(self, problem : Problem, grounder_helper : Optional[GrounderHelper] = None):
        GroundCounter.__init__(self, problem)
        self.get_ground_problem(grounder_helper)    
        
    def get_ground_size(self):
        return len(self.ground_actions)


class GroundCounterMacroExact(GroundCounter):

    def __init__(self, problem : Problem, lifted_macro : LiftedMacroEvent, 
                 grounder: Optional[GrounderHelper] = None):
        GroundCounter.__init__(self, problem)
        self.lifted_macro = lifted_macro
        # print(self.lifted_macro)
        if not grounder:
            self.grounder = GrounderHelper(self.problem, prune_actions=True)
        else:
            self.grounder = grounder
        self._fast_cache = {}   
        # self._grounded_actions = set()
        # self.get_ground_problem()
            
    def get_free_params(self, lifted_action, already_assigned):
        params = []
        for p in lifted_action['variables']:
            if p not in already_assigned:
                params.append(p)

        return params
    
    def covers(self, params_legend: Dict, action: Dict[str, Union[List[Parameter], Action]]):

        for var in action['variables']:
            if var not in params_legend:
                return False
        
        return True

    def get_ground_size_fast(self, depth, so_far: Dict): #recursive algorithm to count the exact number of ground size (faster than get_ground_size)
        
        # caching 
        relevant_so_far = []
        for i in range(depth, len(self.lifted_macro)):
            all_var = self.lifted_macro.actions_variables[i]['variables']
            for var in all_var:
                value = so_far.get(var, None)
                if value is not None:
                    relevant_so_far.append((var, value))
        cache_key = (depth, tuple(relevant_so_far)) 
        cached = self._fast_cache.get(cache_key, None)
        if cached is not None:
            return cached

        #print(f"{depth} : {so_far}")
        if depth == len(self.lifted_macro):
            return 1
        else:
            action = self.lifted_macro.actions_variables[depth]
            if self.covers(so_far, action):
                par = tuple()
                for var in action['variables']:
                    par += (so_far[var],)
                if par in self.grounder._action_to_params[action['action'].name]:
                    res = self.get_ground_size_fast(depth+1, so_far)
                else:
                    res = 0
                self._fast_cache[cache_key] = res
                return res
            else:
                res = 0
                # print(self.grounder._action_to_params[action['action'].name])
                for par in self.grounder._action_to_params[action['action'].name]:
                    # new_so_far = {}
                    new_so_far = so_far.copy() # in so_far there are also assignments that do not appear in this action
                    good = True
                    for i,p in enumerate(action['variables']):
                        obj = so_far.get(p,None)
                        if obj:
                            if par[i] != obj:
                                good = False
                                break         
                        else:
                            new_so_far[p] = par[i]
                    if good:
                        res += self.get_ground_size_fast(depth+1, new_so_far)
                self._fast_cache[cache_key] = res
                return res

    def get_ground_size(self): #count checking if every possible combination of params ground give actually a ground action
        size = 0
        parameter_list = list(self.lifted_macro.variables)
        parameter_indexes = {p:i for i,p in enumerate(parameter_list)}
        all_parameters = get_all_parameters(self.problem, parameter_list)
        
        for par in all_parameters:
            for d in self.lifted_macro.actions_variables:
                act = d['action']
                var = d['variables']
                params = []
                for v in var:
                    ind = parameter_indexes[v]
                    params.append(par[ind])
                key = (act.name, tuple(params))
                ground_act = self.grounder._grounded_actions[key]
                if not ground_act:
                    break
            if ground_act:
                size += 1
        
        return size
        
    def get_ground_size_with_grounding(self):
        return len(get_ground_set(self.lifted_macro, self.problem))

    def get_ground_size_alternative(self): # very slow, for now useless : consider all possible products between ground actions of the lifted and check (params -> obj) if it is the grounded of that lifted 
        self._actions_map : Dict[str, List[str]] = {}
        for old_action in self.problem.actions:
            for grounded_params in self.grounder.get_possible_parameters(old_action):
                new_action = self.grounder.ground_action(old_action, grounded_params)
                if new_action is not None:
                    self._actions_map.setdefault(old_action.name, []).append(new_action.name)

        size = 0
        params = [self._actions_map[act.name] for act in self.lifted_macro.actions]
        all_possible_macros = list(itertools.product(*params))

        for macro in all_possible_macros:
            ma = GroundMacroEvent(macro)
            if self.lifted_macro.check_grounding(ma, tuple([a.name for a in self.lifted_macro.actions])):
                size += 1
        return size

class BestImpactSetMacros(Annealer):
    """ Use annealer to approximate the minimum of expanded states"""

    def __init__(self, database_lifted_macros : List[LiftedMacroEvent], n_actions, len_plan, macros_usage : str, initial_state, 
                 max_macros : Optional[int] = None, to_sort : Optional[List[int]]=None, load_state=None):
        super().__init__(initial_state, load_state)
        self.database_lifted_macros = database_lifted_macros
        self.n_actions = n_actions
        self.len_plan = len_plan
        self.macros_usage = macros_usage
        self.max_macros = max_macros
        self.to_sort = to_sort

    def update(self, *args, **kwargs):
        """Wrapper for internal update.

        If you override the self.update method,
        you can chose to call the self.default_update method
        from your own Annealer.
        """
        #self.default_update(*args, **kwargs)
        pass

    def move(self):
        """ Swaps one value from 0 to 1 or viceversa"""
        if self.to_sort:
            i = random.choice(self.to_sort)
        else:
            i = random.randint(0, len(self.state) - 1)
        if self.state[i] == 0:
            if self.max_macros:
                pivot = random.randint(0, len(self.state) - 1)
                best_guess = None
                total = 0
                for j, x in enumerate(self.state):
                    if x == 1:
                        total += 1
                        if (best_guess is None) or (abs(best_guess - pivot) > abs(j - pivot)):
                            best_guess = j
                if total == self.max_macros:
                    self.state[best_guess] = 0  

            self.state[i] = 1
        else:
            self.state[i] = 0

    def energy(self):
        bf = self.n_actions 
        diff_depth = 0
        max_len = 1

        if '+' in self.macros_usage:
            M = deque((item for flag, item in zip(self.state, self.database_lifted_macros) if flag)) # converting list of {0,1} to the corresponding set of LiftedMacroEvent
            visited = set()
            len_visited = 0
            while M:
                macro = M.popleft()  
                visited.add(macro)
                if len_visited != len(visited): # Check if this is the first time I encounter this macro
                    diff_depth += macro.cumulative_diff_depth_negative
                    max_len = max(max_len, len(macro))  
                    M.append(macro.get_last_sub_macro())
                bf += macro.size_ground_exact
                len_visited = len(visited)
        else:
            for i, flag in enumerate(self.state):
                if flag:
                    macro = self.database_lifted_macros[i]
                    diff_depth += macro.cumulative_diff_depth_negative
                    max_len = max(max_len, len(macro))  
                    bf += macro.size_ground_exact
               
        
        min_depth = Fraction(self.len_plan/max_len)
        depth = max(self.len_plan-diff_depth, min_depth)
        es = expanded_states(depth, bf)
        #print(es)

        return es 

def find_optimal_set(lifted_macros, n_actions, len_plan, macros_usage, max_macros, timeout : Optional[float] = None, to_sort: Optional[List[int]] = None):


    if macros_usage == 'PA++' or max_macros is not None:
        initial_state = [0] * len(lifted_macros)
        initial_state[random.randint(0, len(lifted_macros) - 1)] = 1
    else:
        initial_state = [random.choice([0, 1]) for _ in range(len(lifted_macros))]

    opt = BestImpactSetMacros(lifted_macros, n_actions, len_plan, macros_usage, initial_state=initial_state, max_macros=max_macros, to_sort=to_sort)

    if timeout:
        auto_schedule = opt.auto(minutes=timeout) # {'tmin': ..., 'tmax': ..., 'steps': ...}
        opt.set_schedule(auto_schedule)

    best_set_ind, best_impact = opt.anneal()

    print(f"Macros selected: {best_set_ind}")

    best_set = [lifted_macros[i] for i, status in enumerate(best_set_ind) if status==1]

    return best_set, best_impact
    

def extract_lifted_macros_from_json(database, problem):

    new_database = []
    cumulative_diff_depth_negative = None
    cumulative_diff_depth_positive = None
    size_ground_ma_in_database = None

    factory = LiftedMacroEventFactory()

    for lifted_macro in database:
        ma = []
        for d in lifted_macro:
            if len(d) == 2:
                lifted = {}
                act = problem.action(d['action'])
                lifted['action'] = act
                variables = []
                for i, var in enumerate(d['variables']):
                    ut = act.parameters[i].type
                    par = Parameter(name = var, typename = ut)
                    variables.append(par)
                lifted['variables'] = variables
                ma.append(lifted)
            else:
                if 'cumulative_diff_depth_negative' in list(d.keys()):
                    cumulative_diff_depth_negative = d['cumulative_diff_depth_negative']
                elif 'cumulative_diff_depth_positive' in list(d.keys()):
                    cumulative_diff_depth_positive = d['cumulative_diff_depth_positive']
                else:
                    size_ground_ma_in_database = d['size_ground_ma_in_database']
        lma = factory.make_macro(ma) 
        if cumulative_diff_depth_negative:
            lma.cumulative_diff_depth_negative = Fraction(cumulative_diff_depth_negative)
        if cumulative_diff_depth_positive:
            lma.cumulative_diff_depth_positive = Fraction(cumulative_diff_depth_positive)
        if size_ground_ma_in_database:
            lma.size_ground_ma_in_database = int(size_ground_ma_in_database)
        new_database.append(lma)

    return new_database

def get_all_parameters(problem, parameter_list):
    
    ground_size = 1
    domain_sizes = []
    type_list = []
    for p in parameter_list:
        t = p.type
        type_list.append(t)
        ds = domain_size(problem, t)
        domain_sizes.append(ds)
        ground_size *= ds
    items_list: List[List[FNode]] = []
    for size, type in zip(domain_sizes, type_list):
        items_list.append(
            [domain_item(problem, type, j) for j in range(size)]
        )
    res = itertools.product(*items_list)
    
    return res

# Remove duplicates based on the `actions` attribute
def remove_duplicates(sorted_list):
    seen = set()
    unique_list = []
    for item in sorted_list:
        if tuple(item.actions) not in seen:  # Use tuple for hashability
            seen.add(tuple(item.actions))
            unique_list.append(item)
    return unique_list


def select_best_lifted_macros(lifted_macros_database, problem, macros_usage, len_mean, max_macros, grounder_helper : Optional[GrounderHelper] = None):

    if max_macros:
        max_macros = int(max_macros)
    if len_mean:
        len_mean = Fraction(len_mean)

    ground_counter_problem = GroundCounterProblem(problem, grounder_helper)
    n_actions_mean = ground_counter_problem.get_ground_size()
    grounder_type = 'exact'


    lifted_macros_database = [ma for ma in lifted_macros_database if ma.is_in_grounded_problem(ground_counter_problem.grounder)]

    # compute ground size of each lifted macro
    if grounder_type == 'exact':
        for ma in lifted_macros_database:
            ground_counter = GroundCounterMacroExact(problem, ma, ground_counter_problem.grounder)
            bf = ground_counter.get_ground_size_fast(depth=0, so_far={})
            ma.size_ground_exact = bf   
            # ground_size[ma] = bf

    lifted_macros_database = list(sorted(lifted_macros_database, key=lambda x: x.es(n_actions_mean, len_mean, grounder_type, macros_usage)))

    lifted_macros_database = remove_duplicates(lifted_macros_database) # filter database selecting best arity for every macro
    if max_macros is not None and max_macros == 1:
        selected_lifted_macros = lifted_macros_database[:1]
    else:
        if 'PA++' in macros_usage:
            max_len_macro = max([len(ma) for ma in lifted_macros_database])
            index_to_sort = [i for i, ma in enumerate(lifted_macros_database) if len(ma) == max_len_macro]
            selected_lifted_macros, _ = find_optimal_set(lifted_macros_database, n_actions_mean, len_mean, macros_usage, max_macros=max_macros, timeout=0.2, to_sort=index_to_sort)
        else:
            selected_lifted_macros, _ = find_optimal_set(lifted_macros_database, n_actions_mean, len_mean, macros_usage, max_macros=max_macros, timeout=0.2)

    return selected_lifted_macros
