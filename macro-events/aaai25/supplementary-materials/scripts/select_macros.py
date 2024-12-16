#!/usr/bin/env python3

import argparse
import json
import re
import os
import statistics
import math
import pandas as pd
import itertools

import signal
import threading
import time

# Define a timeout exception
class TimeoutException(Exception):
    pass

# Handler for the timeout
def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out!")


def get_info_macros(log_file):

    n_random_pos, n_random_neg, n_policy_pos, n_policy_neg, len_trace_pos, len_trace_neg = {}, {}, {} ,{}, {}, {}

    with open(log_file) as f:
        lines = f.readlines()[1:]
        for l in lines:
            v = re.split(r'\s*,\s*', l)
            if v[4] == 'True':
                n_random_pos[int(v[0])] = int(v[-3])
                n_policy_pos[int(v[0])] =  int(v[-2])
                len_trace_pos[int(v[0])] = int(v[-1].split("\n")[0])
            else:
                n_random_neg[int(v[0])] =  int(v[-3])
                n_policy_neg[int(v[0])] = int(v[-2])
                len_trace_neg[int(v[0])] = int(v[-1].split("\n")[0])

    return n_random_pos, n_random_neg, n_policy_pos, n_policy_neg, len_trace_pos, len_trace_neg

def get_sub_macros(index, df):

    macro = df.loc[index, 'macros']
    sub_macros=[]

    macro_cut = macro[1:-1].split('  ')

    for j in range(2,df.loc[index, 'length']):
        sub_macro = macro_cut[:j]
        sub_macro = '('+'  '.join(sub_macro)+')'
        sub_index = df.index[df['macros'] == sub_macro].tolist()
        assert len(sub_index) == 1
        sub_index = sub_index[0]
        sub_macros.append((sub_macro, sub_index))

    sub_macros.append((macro, index))
    return sub_macros

def compute_convex_combination(df, sub_macros, sub_diff_depths):

    coeff = []
    value = 0

    _,ind = sub_macros[0]
    max_freq = df.loc[ind, 'positive frequency']

    for j , sub_macro in enumerate(sub_macros[:-1]):
        sub_index = sub_macro[-1]
        _,next_index = sub_macros[j+1]
        coeff = (df.loc[sub_index, 'positive frequency'] - df.loc[next_index, 'positive frequency'])
        assert coeff >= 0
        value += coeff * sub_diff_depths[j]

    _, last_index = sub_macros[-1]
    coeff = (df.loc[last_index, 'positive frequency'])
    value = coeff * sub_diff_depths[-1]
    value = value / max_freq

    return value

def compute_impact(n, df, comb_ma, depth, macros_usage):

    if "-" in macros_usage:
        n_actions = n+len(comb_ma)
    else:
        sum_lengths = 0
        for i in comb_ma:
            sum_lengths += df.loc[i, 'length']
        n_actions = n - len(comb_ma) + sum_lengths

    card = ((n_actions)**(depth+1)-1)/(n_actions - 1)
    return card


def add_utility(l, n, df, tot_plans, macros_usage, uniqueness_impact):
    # Check if the "positive frequency" column exists
    if "positive frequency" not in df.columns:
        raise ValueError("The DataFrame must contain a 'positive frequency' column.")

    # Create the "utility" column by dividing "positive frequency" by the divisor
    df['utility'] = df['positive frequency'] / tot_plans

    for i in df.index.tolist():
        if "FA" in macros_usage:
            df.loc[i, 'diff_depth'] = float(df.loc[i, 'utility']) * (float(df.loc[i, 'length']) -1 )
        else: 
            sub_diff_depths = []
            sub_macros = get_sub_macros(i, df)
            for _, sub_index in sub_macros:
                sub_diff_depth = float(df.loc[sub_index, 'utility']) * (float(df.loc[sub_index, 'length']) -1 )
                sub_diff_depths.append(sub_diff_depth)
            diff_depth = compute_convex_combination(df, sub_macros, sub_diff_depths)
            df.loc[i, 'diff_depth'] = diff_depth

    if "+" in macros_usage: #In + diff_depth is not enough to rank the single impact, so we need to add impact
        for i in df.index.tolist():
            min_depth = l/df.loc[i, 'length']
            _, new_comb = compute_hat_set((i,), df)
            diff_depth = compute_diff_depth(df, new_comb)
            depth = max(l - diff_depth, min_depth)
            assert depth > 0
            if uniqueness_impact:
                comb = new_comb
            else:
                comb = (i,)
            diff_impact = compute_impact(n, df, comb, depth, macros_usage) # real impact value is intercept - diff_impact
            df.loc[i, 'single_impact'] = diff_impact

    return df

# def add_depth_single(df, len_mean, macros_usage):

#     if "FA" in macros_usage:
#         df['depth_m'] = len_mean - df['diff_depth']
#     elif "PA" in macros_usage:
#         df = compute_PA_macros_length(df)

#     return df


def compute_mean_n_actions_majsp(n_min_robots: int, n_max_robots: int, n_min_pallets: int, n_max_pallets: int, n_min_treatments: int, n_max_treatments: int, n_min_positions: int, n_max_positions: int):

        n_actions = []
        for r in range(n_min_robots, n_max_robots+1):
            for b in range(n_min_pallets, n_max_pallets+1):
                for i in range(n_min_treatments, n_max_treatments+1):
                    for pos in range(n_min_positions, n_max_positions+1):
                        for loc in list(itertools.combinations(range(pos), i)):
                            n=0
                            n += r * (i+1) #move
                            n += r #unload
                            n += r*b # load_at_dep
                            n += 2* r * b * i  #make_treat, load
                            n_actions.append(n)

        return statistics.mean(n_actions)

def compute_mean_n_actions_kitting( n_components: int, kit_size: int, n_kit: int, max_kit: int, n_robots: int):

    n_actions = []
    for r in range(1, n_robots+1):
        for l in range(kit_size + 1):
            for comb in itertools.product(range(n_components), repeat=l):
                for i in range(1,n_kit +1):
                    n=0
                    n += r * (n_components+1) * (n_components+1) #move
                    n += r * n_components * n_components * l * kit_size # load
                    n += 2* max_kit #prepare unload, unload
                    n_actions.append(n)

    return statistics.mean(n_actions)


def f_summation(l,n):
    return (n**(l+1) - 1)/(n -1)

def compute_depth_zero(intercept, base, card_macros):
    value = intercept * (base + card_macros - 1) + 1
    return math.log(value, base+card_macros) - 1


def compute_hat_set(comb, df):

    new_comb = []
    ma_hat = []

    for i in comb:
        #macro = df.loc[i, 'macros']
        sub_macros = get_sub_macros(i, df)
        for sub_macro,ind in sub_macros:
            if sub_macro not in ma_hat:
                ma_hat.append(sub_macro)
                new_comb.append(ind)

    return ma_hat, tuple(new_comb)


def compute_diff_depth(df, comb_ma):
    value = df.loc[comb_ma[0], 'diff_depth']
    if len(comb_ma)>1:
        for j in comb_ma[1:]:
            value += df.loc[j, 'diff_depth']

    return value

def evaluate_set_macros(intercept, l, n, df, max_ma, macros_usage, max_len_macro, uniqueness_int):

    MA_impact = {}
    MA_depth = {}
    combinations = {}

    if "-" in macros_usage:
        for i in range(1, max_ma):
            comb = df.index.tolist()[:i]
            ma = tuple([df.loc[j,'macros'] for j in comb])
            max_len_macro = max([df.loc[j, 'length'] for j in comb])
            min_depth = l/max_len_macro
            diff_depth = compute_diff_depth(df, comb)
            depth = max(l - diff_depth, min_depth)
            assert depth > 0
            impact = compute_impact(n, df, comb, depth, macros_usage)
            if intercept - impact > 0:
                MA_depth[ma] = [depth,i]
                MA_impact[ma] = [impact]

    elif "+" in macros_usage:
        df_new = df
        if 'PA' in macros_usage:
            df_new=df[df['length'] == 5]
        #print(f'Macros with len=5 :  {len(df_new.index.tolist())}')
        df_cut = df_new.iloc[:max_ma] #cut the macros
        #print(f"Indici best macros:  {df_cut.index.tolist()}")
        for i in range(1, max_ma+1):
            for comb in itertools.combinations(df_cut.index.tolist(),i):
                for j in comb:
                    assert df.loc[j,'macros'] == df_cut.loc[j,'macros']
                ma = tuple([df.loc[j,'macros'] for j in comb])
                min_depth = l/max_len_macro
                ma_hat, new_comb = compute_hat_set(comb, df)
                diff_depth = compute_diff_depth(df, new_comb)
                depth = max(l - diff_depth, min_depth)
                assert depth > 0
                if uniqueness_int:
                    comb = new_comb
                impact = compute_impact(n, df, comb, depth, macros_usage) # real impact value is intercept - impact
                if intercept - impact > 0:
                    MA_depth[ma] = [depth,i]
                    MA_impact[ma] = impact
                    combinations[ma] = comb

    return MA_impact, MA_depth, combinations

def evaluate_impact(df, candidate, max_len_macro, macros_usage, l, n, uniqueness_int):

    if 'PA+' == macros_usage:
        max_len = max_len_macro
    elif "FA+" == macros_usage:
        max_len = max([df.loc[k,'length'] for k in candidate])

    min_depth = l/max_len
    _, new_comb = compute_hat_set(candidate, df)
    diff_depth = compute_diff_depth(df, new_comb)
    depth = max(l - diff_depth, min_depth)
    assert depth > 0
    if uniqueness_int:
        candidate = new_comb
    diff_impact = compute_impact(n, df, candidate, depth, macros_usage) # real impact value is intercept - diff_impact
    #impact = intercept - diff_impact

    #return impact
    return diff_impact


def compute_best_set_timeout(df, l, n, intercept, max_len_macro, macros_usage, best_set_plus, uniqueness_int):

    df_new = df
    if 'PA+' == macros_usage:
        df_new=df[df['length'] == max_len_macro]
        #print(f'Macros with len={max_len_macro} :  {len(df_new.index.tolist())}\n')

    total = df_new.index.tolist()
    candidates = []
    i=1
    best_impact = float('inf')
    best_comb = 0

    for j, x in enumerate(total):
        c = [x]
        new_candidates = [c]
        i+= 1
        impact = evaluate_impact(df, c, max_len_macro, macros_usage, l, n, uniqueness_int)
        if impact < best_impact and intercept - impact > 0:
            best_comb = c
            best_impact = impact
            best_set_plus.append(tuple([df.loc[j,'macros'] for j in c]))
            print(f"{i}:{c} find new best")
        for o in candidates:
            c = o + [x]
            new_candidates.append(c)
            i+= 1
            impact = evaluate_impact(df, c, max_len_macro, macros_usage, l, n, uniqueness_int)
            if impact < best_impact and intercept - impact > 0:
                best_comb = c
                best_impact = impact
                best_set_plus.append(tuple([df.loc[j,'macros'] for j in c]))
                print(f"{i}:{c} find new best")
        print(f"{i}:{best_comb} (up to {j+1} macros)")
        candidates.extend(new_candidates)

# Function to run another function with a timeout
def run_with_timeout(func, timeout, *args):
    thread = threading.Thread(target=func, args=args)
    thread.start()

    thread.join(timeout)

    if thread.is_alive():
        print(f"Function timed out after {timeout} seconds!")
    else:
        print("Function completed within the timeout.")

def extract_macros(input_string, timeout):

    if timeout is None:
        input_string = input_string[2:-2].split('", "')

    # Create a DataFrame
    df = pd.DataFrame(input_string, columns=['macros'])

    return df


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)
    parser.add_argument('-j', required=True, type=str)
    parser.add_argument('-o', required=True, type=str)
    parser.add_argument('-u', required=True, type=str)
    parser.add_argument('-d', required=True, type=str)
    parser.add_argument('-m', type=int, default=5)
    parser.add_argument('-t', type=int, default=None)
    parser.add_argument('-n', type=bool, default=False)

    args, _ = parser.parse_known_args()

    input = args.i
    learning_log = args.j
    output = args.o
    macros_usage = args.u
    domain = args.d
    max_len_macro = float(args.m)
    timeout = args.t
    uniqueness_int = args.n
    macros = {}

    if not os.path.exists(output):
        os.mkdir(output)

    _, _, _, _, len_trace_pos, _ = get_info_macros(learning_log)
    len_mean = statistics.mean(list(len_trace_pos.values()))
    tot_plans = len(len_trace_pos)

    if domain == "majsp":
        n_actions_mean = compute_mean_n_actions_majsp(1,4, 1,3, 1,3, 6,6)
    else:
        n_actions_mean = compute_mean_n_actions_kitting(3,4,2,2,2)

    card_T_empty = f_summation(len_mean, n_actions_mean)

    #print(f"L_mean:  {len_mean},  N_mean:   {n_actions_mean},  Tot_plans:  {tot_plans}")

    macros = pd.read_csv(input)
    #print(f"CMA: {macros.shape[0]}")

    macros = macros[macros['positive frequency'] > 0]

    macros = add_utility(len_mean, n_actions_mean, macros, tot_plans, macros_usage, uniqueness_int)

    if '-' in macros_usage:
        sorted_macros = macros.sort_values(by="diff_depth", ascending=False)
    else:
        sorted_macros = macros.sort_values(by="single_impact")
    #sorted_macros.to_csv(os.path.join(output, "sorted_macros.csv"), index=False)

    if "+" in macros_usage:
        max_macros = 12
    else:
        max_macros = 500

    if timeout is None:
        set_macros_impact, set_macros_depth, comb = evaluate_set_macros(card_T_empty, len_mean, n_actions_mean, sorted_macros, max_macros, macros_usage, max_len_macro, uniqueness_int)
        data = {'set_macros': list(set_macros_impact.keys()), 'impact': list(set_macros_impact.values()), 'depth': [v[0] for v in list(set_macros_depth.values())], 'card_set':  [v[1] for v in list(set_macros_depth.values())]}
        set_macros_df = pd.DataFrame(data)
        #print(f"Length of set macros:  {set_macros_df.shape[0]}")
        sorted_set_macros = set_macros_df.sort_values(by='impact')
        #if "+" in macros_usage:
            #print(f"Indici best set:   {comb[sorted_set_macros.loc[sorted_set_macros.index.tolist()[0],'set_macros']]}")
        #sorted_set_macros.to_csv(os.path.join(output, 'set_macros.csv'), index=False)
        best_set = extract_macros(str(sorted_set_macros.loc[sorted_set_macros.index.tolist()[0],'set_macros']), timeout)
    else:
        assert "+" in macros_usage
        best_set_plus = []
        run_with_timeout(compute_best_set_timeout, timeout, sorted_macros, len_mean, n_actions_mean, card_T_empty, max_len_macro, macros_usage, best_set_plus, uniqueness_int)
        best_set = best_set_plus[-1]
        best_set = extract_macros(best_set, timeout)

    best_set.to_csv(os.path.join(output, 'best_macros_{}.csv'.format(macros_usage)), index=False)

    if timeout is not None:
        os._exit(0)

if __name__ == '__main__':
    main()
