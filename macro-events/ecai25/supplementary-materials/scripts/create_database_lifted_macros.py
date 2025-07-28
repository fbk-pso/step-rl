#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import math
import pandas as pd
import itertools
from typing import Tuple, Dict, List, Optional
from collections import OrderedDict
import json
import re
from scripts.utils import GroundMacroEvent, GroundMacroEventFactory, LiftedMacroEvent, LiftedMacroEventFactory
from unified_planning.shortcuts import *

from unified_planning.engines.compilers import Grounder, GrounderHelper
from unified_planning.io import ANMLReader
from unified_planning.model.action import Action
from unified_planning.plans.plan import ActionInstance
from unified_planning.model.parameter import Parameter
from fractions import Fraction

LIFTED_FACTORY = LiftedMacroEventFactory()

N_ACTIONS_KITTING = 34 #max kit_size 4
N_ACTIONS_MAJSP = 21.5


def partition(collection, only_one):
    if len(collection) == 1 or only_one:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:], only_one):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def generate_all_arity_combinations(input, only_one):

    res = None
    for col in input:
        tmp = []
        for p in partition(col, only_one):
            if res is None:
                tmp.append([p])
            else:
                for x in res:
                    tmp.append(x+[p])
        res = tmp

    for i, x in enumerate(res):
        res[i] = list(itertools.chain.from_iterable(x))

    return res


def find_action_instance(ga : str, ground_problem, ground_actions_name, map_back_ai) -> Action:

    i = ground_actions_name.index(ga)
    ground_ai = ActionInstance(ground_problem.actions[i], tuple()) 
    lifted_ai = map_back_ai(ground_ai)

    return lifted_ai

def extract_lifted_macros(me: GroundMacroEvent, ground_problem, map_back_ai, only_one : bool): #only one return just the lifted with the short number of parameters

    #extract actions and objects from all actions
    ground_actions_name = [gr_act.name for gr_act in ground_problem.actions]
    actions : Dict[ActionInstance, List[int]] = {} #ActionInstance because in this way two different instances of same action are considered different
    all_objects = []
    i = 0
    for ga in me.actions:
        ai = find_action_instance(ga, ground_problem, ground_actions_name, map_back_ai)
        if ai.action not in set(actions.keys()):
            actions[ai] = []
        actions[ai].extend([i + j for j in range(len(ai.actual_parameters))])
        i += len(ai.actual_parameters)
        all_objects.append(list(ai.actual_parameters))
    all_objects = list(itertools.chain.from_iterable(all_objects))


    # generate all possible combinations of lifted macro (based on computing partitions of indeces of equal objects)
    equivalence_class_index = []
    for obj in set(all_objects):
        equivalence_class_index.append([i for i, x in enumerate(all_objects) if x == obj])
    all_combinations = generate_all_arity_combinations(equivalence_class_index, only_one)

    # for every partition extract the lifted macro
    for partitions in all_combinations:
        # print(f"{macro_string} : {partitions}")
        obj_to_var = {}
        for i, l in enumerate(partitions):
                ty = all_objects[l[0]].type
                var = Parameter(name = 'v'+str(i), typename = ty)
                for j in l:
                    obj_to_var[j] = var           
        lifted_macro = []
        for act, ind in actions.items(): 
            lifted_action = {'action' : None, 'variables' : None}
            assert isinstance(act.action, Action)
            lifted_action['action'] = act.action
            variables = []
            for k in ind:
                if 'integer' in str(obj_to_var[k].type):
                    for p in act.action.parameters:
                        if 'integer' in str(p):
                            ty = str(p).split('] ')[0] + ']'
                            match = re.search(r'integer\[(\d+), (\d+)\]', ty)
                            lb = int(match.group(1))
                            upb = int(match.group(2))
                            ut = IntType(lb,upb)
                            var = Parameter(name = obj_to_var[k].name, typename = ut)
                            obj_to_var[k] = var
                            break
                variables.append(obj_to_var[k])
            lifted_action['variables'] = variables
            lifted_macro.append(lifted_action)
        yield lifted_macro        


def create_database_lifted_macros(gme_dict: Dict[Tuple[str], List[GroundMacroEvent]], ground_problem, map_back_ai):
    """
    Create database of lifted macros. 
    
    Return a dict whose keys are lifted macros and whose values are cumulative diff_depth
    """
    # grounder_helper = GrounderHelper(problem, prune_actions=False) 
    es = {} # expanded states
    j = 1

    for family, gme in gme_dict.items():
        for me in gme:
            k = 0
            n = 0
            for ma in extract_lifted_macros(me, ground_problem, map_back_ai, only_one=True):
                lifted_macro = LIFTED_FACTORY.make_macro(ma)
                tmp = es.get(lifted_macro, [0,0]) 
                if tmp == [0,0]:
                    n += 1
                    for ground_macro in gme:
                        if lifted_macro.check_grounding(ground_macro, family=family):
                            tmp[0] += ground_macro.diff_depth
                            # print(f"{type(ground_macro.diff_depth.numerator)} - {type(ground_macro.diff_depth.numerator)}")
                            tmp[1] += 1
                    es[lifted_macro] =  tmp
                k+=1
            #print(f"{j})  {me}  :  found -> {k},  new -> {n}")
            j+=1

    return es

def extract_macro_family(macro : GroundMacroEvent, ground_problem, ground_actions_name, map_back_ai):

    macro_family = []

    for gma in macro.actions:
        ai = find_action_instance(gma, ground_problem, ground_actions_name, map_back_ai)
        macro_family.append(ai.action.name)

    return tuple(macro_family)


def add_utility_and_separate(df, tot_plans, macros_usage, ground_problem, map_back_ai):

    macros = {} # dict -> family : macros
    ground_actions_name = [gr_act.name for gr_act in ground_problem.actions]

    # Check if the "positive frequency" column exists
    if "positive frequency" not in df.columns:
        raise ValueError("The DataFrame must contain a 'positive frequency' column.")
    
    # l_max = df['length'].max()
    
    # if 'PA+' == macros_usage:
    #     database = df[df['length'] == l_max]
    # else:
    database = df.sort_values(by='length', ascending=True)
    factory = GroundMacroEventFactory()

    for i in database.index.tolist():
        macro = factory.make_macro(df.loc[i, 'macros'])
        macro_family = extract_macro_family(macro, ground_problem, ground_actions_name, map_back_ai)
        macros.setdefault(macro_family, []).append(macro)
        # if macro_family in set(macros.keys()):
        #     macros[macro_family].append(macro)
        # else:
        #     macros[macro_family] = [macro]
        macro.positive_frequency = df.loc[i, 'positive frequency']
        macro.length = df.loc[i, 'length']
        macro.utility = Fraction(int(macro.positive_frequency), int(tot_plans))
        macro.diff_depth = macro.utility * (len(macro) - 1 )
        if "PA" in macros_usage and len(macro) > 2: 
            sub_macros = [s for s in macro.get_sub_macros()]
            sub_macros.append(macro)
            denominator = sub_macros[0].utility 
            macro.diff_depth = macro.diff_depth * Fraction(macro.utility / denominator)
            for i, sub_macro in enumerate(sub_macros[:-1]):
                numerator = sub_macro.utility - sub_macros[i+1].utility
                coeff = Fraction(numerator / denominator)
                macro.diff_depth += coeff * sub_macro.diff_depth
    
    # Sort the dictionary macros according to the length of keys (keys are tuples)
    sorted_macros = OrderedDict(sorted(macros.items(), key=lambda item: len(item[0])))

    return sorted_macros

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str, help='Database of ground macros file .csv') 
    parser.add_argument('-j', required=True, type=str, help='database_valid_plans') 
    parser.add_argument('-o', required=True, type=str, help='Output directory')  
    parser.add_argument('-u', required=True, type=str, help="Macros usage: 'FA-', 'FA+', 'PA-', 'PA+'")
    # parser.add_argument('-d', required=True, type=str, help="Benchmark: 'majsp', 'kitting'")
    parser.add_argument('-p', required=True, type=str, help='Path to domain.anml')

    args, _ = parser.parse_known_args()

    input = args.i
    database_valid_plans = args.j
    output = args.o
    macros_usage = args.u
    # domain = args.d
    problem_path = args.p

    if not os.path.exists(output):
        os.makedirs(output)

    r = ANMLReader()
    problem = r.parse_problem(problem_path)
    # take ground problem to recover lifted action of original problem
    grounder = Grounder(prune_actions=False)
    grounding_result = grounder.compile(problem)
    ground_problem = grounding_result.problem
    map_back_ai = grounding_result.map_back_action_instance

    # read and compute average length of plans and average number of ground actions
    database_valid_plans = pd.read_csv(database_valid_plans)
    tot_plans = database_valid_plans.shape[0]

    # read ground macros
    ground_macros_df = pd.read_csv(input)
    ground_macros = add_utility_and_separate(ground_macros_df, tot_plans, macros_usage, ground_problem, map_back_ai) # dict: {sequence of actions : list of ground macros with same sequence of actions}


    print("\nStarting creating database of lifted macros...\n")

    #create database of lifted macros
    lifted_macros_database = create_database_lifted_macros(ground_macros, ground_problem, map_back_ai) #return a dict whose keys are lifted macros
    print(f'Total number of lifted macros:  {len(lifted_macros_database)}')

    lme = {} # lifted macro : cumulative_diff_depth_positive

    for ma, es in lifted_macros_database.items():
        ma.cumulative_diff_depth_negative = es[0]
        ma.cumulative_diff_depth_positive = ma.cumulative_diff_depth_negative
        if '+' in macros_usage :
            if len(ma) > 2:
                sub_ma = ma.get_last_sub_macro()
                ma.cumulative_diff_depth_positive += sub_ma.cumulative_diff_depth_positive  # correct because the lifted_macros_database is sorted according to the length of the macro
        ma.size_ground_ma_in_database = es[1]


    #write database on file json
    lma_to_json = [lma.to_json() for lma in lifted_macros_database.keys()]
    with open( os.path.join(output, 'database_lifted_macros_{}.json'.format(macros_usage)), "w" ) as write:
        json.dump( lma_to_json , write , indent = 4)

if __name__ == '__main__':
    main()
