import os
import pandas as pd
import argparse
from typing import Dict, Set
import statistics


def find_frequent_macros(CME: Dict, trace: tuple, max_len: int):


    for i in range(1, max_len):

        window_start = 0
        window_end = i+1

        while window_end <= len(trace):
            macro = trace[window_start:window_end]
            window_start += 1
            window_end += 1
            frequency = CME.get(macro, 0)
            CME[macro] = frequency + 1

def select_shortest_traces(df: pd.DataFrame, val: int) -> pd.DataFrame:
    """
    Reduces the input DataFrame by selecting the first `val` rows with the lowest 
    'len_trace' value for each unique 'problem_id'.

    Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'problem_id' and 'len_trace'.
        val (int): The number of rows to keep for each unique 'problem_id'.

    Returns:
        pd.DataFrame: The reduced DataFrame.
    """
    # Group by 'problem_id' and sort each group by 'len_trace'
    grouped = (
        df.sort_values(by=[' problem_id', 'len_trace'])
          .groupby(' problem_id', group_keys=False)
    )
    
    reduced_df = grouped.head(val)
    
    return reduced_df



def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str, help='Database of traces file .csv') 
    parser.add_argument('-o', required=True, type=str, help='Output directory')
    parser.add_argument('-m', required=True, type=str, help='Max length macros')
    parser.add_argument('-t', type=str, default=None, help='Number of traces for every instances selected')

    args, _ = parser.parse_known_args()

    input = args.i
    output = args.o
    max_len = int(args.m)
    max_traces = args.t
    if max_traces:
        max_traces = int(max_traces)

    if not os.path.exists(output):
        os.mkdir(output)

    database_traces = pd.read_csv(input)
    database_valid_plans = database_traces[database_traces[' solved'] == ' True']
    l = database_valid_plans.shape[0]
    database_valid_plans = database_valid_plans.drop_duplicates(subset=[' problem_id', ' solved', ' trace'], keep='first')
    print(f"Duplicates traces removed : {l - database_valid_plans.shape[0]}")
    print(f"Total number of valid plans : {database_valid_plans.shape[0]}")

    print("\nStarting extracting CME ...\n")

    CME = {} #dict macro : frequency

    for j in database_valid_plans.index.to_list():
        t = database_valid_plans.loc[j, ' trace']
        trace = tuple(t.split('|'))
        database_valid_plans.loc[j, 'len_trace'] = len(trace)
        if max_traces is None:
            find_frequent_macros(CME, trace, max_len)
    
    if max_traces:
        database_selected_valid_plans = select_shortest_traces(database_valid_plans, max_traces)
        for j in database_selected_valid_plans.index.to_list():
            t = database_selected_valid_plans.loc[j, ' trace']
            t = t[1:] #the first is a space (to correct in the future)
            trace = tuple(t.split('|'))
            find_frequent_macros(CME, trace, max_len)

    print(f"Final number of valid plans : {database_selected_valid_plans.shape[0]}")
    
    database_selected_valid_plans.to_csv(os.path.join(output,'database_valid_plans.csv'), index=False)


    with open(os.path.join(output, 'database_CME.csv'), 'wt') as datafile:
        datafile.write('macros,positive frequency,length\n')
        for macro, freq in CME.items():
            datafile.write('{},{},{}\n'.format(' '.join(str(macro).split(',')), freq, len(macro)))

    print(f"Number of CME : {len(CME)}")
    print(f"Average length of plans : {round(statistics.mean(list(database_selected_valid_plans['len_trace'])),2)}")
    
if __name__ == '__main__':
    main()
