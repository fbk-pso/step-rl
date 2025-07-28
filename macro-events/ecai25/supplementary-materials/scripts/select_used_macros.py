#!/usr/bin/env python3

import argparse
import os
import json
import pandas as pd
from typing import List, Set, Tuple
from scripts.create_database_lifted_macros import extract_lifted_macros 
from scripts.utils import (
    LiftedMacroEvent, 
    GroundMacroEventFactory, 
    GroundMacroEvent, 
    extract_lifted_macros_from_json 
)
from unified_planning.shortcuts import *
from unified_planning.engines.compilers import Grounder
from unified_planning.io import ANMLReader

def extract_ground_macros(used_macros):
    """
    Extracts ground macros and frequency from the list of used macros in every instance.
    """

    # ground_macros : List[GroundMacroEvent] = [] 
    ground_factory = GroundMacroEventFactory()

    for i, macro_list in enumerate(used_macros):
        macros = macro_list.split("|")
        for ma in macros:
            ma = ma.split(":")
            macro = ma[0].strip("()").split(", ")
            macro = [a.strip("'") for a in macro]
            macro = ground_factory.make_macro(macro)
            macro.validation_frequency += int(ma[1]) # Frequency
            macro.validation_support.add(i) # Instance
            # ground_macros.append(macro)

    return list(ground_factory.macros_memory.values())

def create_lifted_macros(ground_macros: List[GroundMacroEvent], ground_problem, map_back_ai, factory):
    """
    Creates lifted macros from the list of ground macros.
    """

    lifted_macros : Set[LiftedMacroEvent] = set()

    for ground_macro in ground_macros:
        for ma in extract_lifted_macros(ground_macro, ground_problem, map_back_ai, only_one=True): # It is not a for with only_one
            lifted_macro = factory.make_macro(ma)
            lifted_macros.add(lifted_macro)
            # if lifted_macro.actions_variables[1] == lifted_macro.actions_variables[-1] and len (lifted_macro) == 5:
            #     print(lifted_macro, ground_macro)
            lifted_macro.validation_frequency += ground_macro.validation_frequency
            lifted_macro.validation_support.update(ground_macro.validation_support)

    return lifted_macros       


def select_lifted_macros_with_cumulative_percentage(lifted_macros: List, thresholds: Tuple[float], total_instances: int) -> Set:
    index = 0
    selected_macros = set()

    sorted_macros_frequency = sorted(lifted_macros, key=lambda macro: macro.validation_frequency, reverse=True)
    # Select the top 70% of the macros based on validation frequency
    lst = [macro.validation_frequency for macro in sorted_macros_frequency]
    threshold = thresholds[0]
    total = sum(lst)
    running_sum = 0
    for i, val in enumerate(lst):
        running_sum += val
        if running_sum / total >= threshold:
            index = i + 1  # because index starts from 0
            break
    selected_macros.update(set(sorted_macros_frequency[:index]))   

    index = 0
    sorted_macros_support = sorted(lifted_macros, key=lambda macro: len(macro.validation_support), reverse=True)
    # Select the top 70% of the macros based on validation support
    threshold = thresholds[1]
    for i in range(len(sorted_macros_support)):
        support = len(set.union(*[ma.validation_support for ma in sorted_macros_support[:i+1]]))
        if support / total_instances >= threshold:
            index = i + 1  # because index starts from 0
            break
    new_elements = set(sorted_macros_support[:index]) - selected_macros 
    print(f"{len(new_elements)} new elements : {new_elements}")
    selected_macros.update(set(sorted_macros_support[:index]))   

    return selected_macros

def select_lifted_macros_ranking(lifted_macros: List, threshold : Tuple[float], total_instances: int) -> Set:
    selected_macros = set()
    total_frequencies = sum([macro.validation_frequency for macro in lifted_macros])

    for macro in lifted_macros:
        if macro.validation_frequency / total_frequencies >= threshold[0] or len(macro.validation_support) / total_instances >= threshold[1]:
            selected_macros.add(macro)

    return selected_macros

def remove_sub_macros(lifted_macros: Set[LiftedMacroEvent]) -> Set[LiftedMacroEvent]:
    total = len(lifted_macros)
    for macro in lifted_macros.copy():
        if len(macro) > 2:
            for sub_macro in macro.get_sub_macros():
                lifted_macros.discard(sub_macro)
    
    print(f"Removed {total - len(lifted_macros)} sub-macros")

    return lifted_macros

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', required=True, help="Path to the input CSV file")
    parser.add_argument('-j', required=True, type=str, help='Complete database of lifted macros file .json') 
    parser.add_argument('-p', required=True, type=str, help='Path to domain.anml')
    parser.add_argument('-o', required=True, type=str, help='Output directory')
    parser.add_argument('-u', required=True, type=str, help="Macros usage: 'FA_minus', 'FA_plus', 'PA_minus', 'PA_plus'")
    parser.add_argument('-t', required=True, type=str, help='Threshold for selection')

    args, _ = parser.parse_known_args()

    input = args.input
    path_lifted_macros = args.j
    problem_path = args.p
    output = args.o
    macros_usage = args.u
    threshold = tuple([float(s) for s in args.t.split(",")])

    if not os.path.exists(output):
        os.makedirs(output)

    # Read the complete database of lifted macros
    with open(path_lifted_macros, 'r') as file:
        database_read = json.load(file)

    r = ANMLReader()
    problem = r.parse_problem(problem_path)
    # take ground problem to recover lifted action of original problem
    grounder = Grounder(prune_actions=False)
    grounding_result = grounder.compile(problem)
    ground_problem = grounding_result.problem
    map_back_ai = grounding_result.map_back_action_instance

    complete_db_lifted_macros = extract_lifted_macros_from_json(database_read, problem)
    factory = complete_db_lifted_macros[0].factory

    # Read the CSV file and extract ground macros used
    df = pd.read_csv(input)
    df = df[df['parameters'].str.contains(macros_usage, na=False)]
    used_macros = df['Macros used'].dropna().tolist() # Extract the 'Macros used' column, drop NaN, and convert to list
    total_instances = len(used_macros)
    print(f"Total instances : {total_instances}")
    ground_macros = extract_ground_macros(used_macros) 

    lifted_macros = create_lifted_macros(ground_macros, ground_problem, map_back_ai, factory)
    print(f"Total used macros : {len(lifted_macros)}")

    macros_usage = macros_usage.replace('_minus', '-').replace('_plus', '+')
    # Select lifted macros
    best_lifted_macros = select_lifted_macros_with_cumulative_percentage(list(lifted_macros), threshold, total_instances)
    # best_lifted_macros = select_lifted_macros_ranking(list(lifted_macros), threshold, total_instances)

    if macros_usage == 'PA+':
        best_lifted_macros = remove_sub_macros(best_lifted_macros)

    for i, ma in enumerate(best_lifted_macros):
        print(f"{i+1}) {ma}, {ma.validation_frequency}, {len(ma.validation_support)}")
    
    # Write database on file json
    lma_to_json = [lma.to_json(planning=True) for lma in best_lifted_macros]
    with open(os.path.join(output, 'best_lifted_macros_{}.json'.format(macros_usage)), "w" ) as write:
        json.dump( lma_to_json , write , indent = 4)

if __name__ == '__main__':
    main()