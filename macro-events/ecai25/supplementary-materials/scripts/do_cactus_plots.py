import matplotlib.pyplot as plt
import argparse
import re
import json
from collections import deque
import os
from os import listdir
from os.path import abspath
import pandas as pd
import itertools

import matplotlib as mpl
mpl.rcParams['pgf.texsystem'] = 'pdflatex' 
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np
from tabulate import tabulate


LINES = {'no_macros'    : '-',
         '(hff, no_macros)' : '--',
         'FA_minus'     : '--',
         'FA_plus'      : '-.',
         'PA_minus'     : '-',
         'PA_plus'      : '--',
         '(hadd, FA_minus)' : '-',
         '(hadd, FA_plus)' : '-',
         '(hadd, PA_minus)' : '-',
         '(hadd, PA_plus)' : '-',
         '(hff, FA_minus)' : '--',
         '(hff, FA_plus)' : '--',
         '(hff, PA_minus)' : '--',
         '(hff, PA_plus)' : '--',
         'best_solver' : '-.',
         '(, best_solver)' : '-.',
         '0.5'        : '-',
         '0.6'        : '--',
         '0.7'        : '-.',
         '0.8'        : '-',
         '0.9'        : '--',
         '1.0'          : '-.'}

COLORS = {'no_macros'   : 'k',
          '(hff, no_macros)' : 'k',
          'FA_minus'    : 'y',
          'FA_plus'     : 'b',
          'PA_minus'    : 'g',
          'PA_plus'     : 'c',
          '(hadd, FA_minus)' : 'y',
          '(hadd, FA_plus)' : 'b',
          '(hadd, PA_minus)' : 'g',
          '(hadd, PA_plus)' : 'c',
          '(hff, FA_minus)' : 'y',
          '(hff, FA_plus)' : 'b',
          '(hff, PA_minus)' : 'g',
          '(hff, PA_plus)' : 'c',
          'best_solver' : 'orange',
          '(, best_solver)' : 'orange',
          'smtinc'      : 'r',
          'smtsua'      : 'm',
          'painter'     : 'k',
          'majsp'       : 'y',
          'tms'         : 'b',
          'satellite'   : 'g',
          'mapanalyser' : 'c',
          'driverlog'   : 'r',
          'matchcellar' : 'm',
          'floortile'   : 'gray',
          'tamer'       : 'r',
          'optic'       : 'b',
          '0.5'         : 'k',
          '0.6'         : 'y',
          '0.7'         : 'b',
          '0.8'         : 'g',
          '0.9'         : 'c',
          '1.0'           : 'r'}

MARKERS = {'painter'     : 'o',
           'majsp'       : 'v',
           'tms'         : 'P',
           'satellite'   : 'D',
           'mapanalyser' : '*',
           'driverlog'   : '^',
           'matchcellar' : 'x',
           'floortile'   : 'p',
           'tamer'       : 'P',
           'optic'       : '^'}

LABELS = {'no_macros' : 'no macros',
          '(hff, no_macros)' : 'no macros',
          'FA_minus' : 'FA-',
          'FA_plus' : 'FA+',
          'PA_minus' : 'PA-',
          'PA_plus' : 'PA+',
          'best_solver' : 'Best Solver',
          '(, best_solver)' : 'Best Solver',
          '(hadd, FA_minus)' : '(hadd, FA-)',
          '(hadd, FA_plus)' : '(hadd, FA+)',
          '(hadd, PA_minus)' : '(hadd, PA-)',
          '(hadd, PA_plus)' : '(hadd, PA+)',
          '(hff, FA_minus)' : '(hff, FA-)',
          '(hff, FA_plus)' : '(hff, FA+)',
          '(hff, PA_minus)' : '(hff, PA-)',
          '(hff, PA_plus)' : '(hff, PA+)',
          '0.5' : '0.5',
          '0.6' : '0.6',
          '0.7' : '0.7',
          '0.8' : '0.8',
          '0.9' : '0.9',
          '1.0' : '1'}

LATEX_LABELS = {'painter'     : '\\painter',
                'majsp'       : '\\majsp',
                'tms'         : '\\TMS',
                'satellite'   : '\\Satellite',
                'mapanalyser' : '\\MAP',
                'driverlog'   : '\\Driverlog',
                'matchcellar' : '\\Matchcellar',
                'floortile'   : '\\Floortile',
                'tamer'       : '\\tamer',
                'optic'       : '\\optic',
                'mono'        : '\\monostn',
                'delta'       : '\\deltastn',
                'clone'       : '\\clonestn',
                'lp'          : '\\linprog',
                'smtmono'     : '\\smt',
                'smtinc'      : '\\smtinc',
                'smtsua'      : '\\smtsua'}


CACTUS_EXCLUDE = ['clone']

FONTSIZE=19

LABELSPACING=0.18


class FakeLabels:
    def __getitem__(self, x):
        return x

def extract_usage(x):
    if 'FA_minus' in x:
        return 'FA_minus'
    elif 'PA_minus' in x:
        return 'PA_minus'
    elif 'PA_plus' in x:
        return 'PA_plus'
    elif 'FA_plus' in x:
        return 'FA_plus'
    else:
        return 'no_macros'

def extract_time(x):
    if x < 1:
        return 1
    elif x > 599:
        return 600
    else:
        return x


def plot_cactus(df, quantity, cumulative=False, title=None, show=False, fname=None, logscale=False, unit=None, labels=FakeLabels(), groupby=None):
    plt.figure(figsize=(10, 6))
    plt.gca().margins(0.02)

    df = df[df["plan-found"]==True]

    if groupby[0] == 'threshold':
        df = df[df['threshold'] != 'baseline']

    tmp = {}
    for solver, gdata in df.groupby(groupby):
        if solver not in CACTUS_EXCLUDE:
            # if len(solver) == 1:
            #     s = solver[0]
            #     majsp = False
            # else:
            s = '('+', '.join(solver)+')'
            tmp[s] = gdata[quantity].sort_values()
            # print("%s : %d" % (s, len(tmp[s])))
            if cumulative:
                tmp[s] = tmp[s].cumsum() 

    # for solver, _ in sorted(tmp.items(), key=lambda x: (len(x[1]), ord(x[0][0]))):
    for solver, _ in sorted(tmp.items(), key=lambda x: ord(x[0][1]), reverse=True):
        plot = plt.plot
        if logscale:
            plt.yscale('log')

        label = solver
        plot(range(0, len(tmp[label])),
             np.array(tmp[label]),
             LINES[label],
             color=COLORS[label],
             label=labels[label])

    plt.xlabel("Instances Solved", fontsize=FONTSIZE)

    lbl = quantity
    if unit is not None:
        lbl += "(%s)" % unit
    if cumulative:
        plt.ylabel("Cumulated Solving %s" % lbl, fontsize=FONTSIZE)
    else:
        plt.ylabel("Solving %s" % lbl, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    if title == 'MAJSP':
        # plt.legend(loc="upper right", frameon=False, prop={'size': 16},
        #        labelspacing=LABELSPACING, ncol =2)
        handles, labels = plt.gca().get_legend_handles_labels()
        # Reorder to get row-wise behavior
        order = [0, 1, 2, 3, 8, 4, 5, 6, 7]  # Put 5 in 2nd col, 4 in 1st
        plt.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="upper right", frameon=False, prop={'size': 16},
                labelspacing=LABELSPACING, ncol=2)
    # else:
    #     plt.legend(loc="lower right", frameon=False, prop={'size': FONTSIZE},
    #            labelspacing=LABELSPACING)
    if title is not None:
        plt.title(title, fontsize=25, weight='bold')

    plt.tight_layout()

    if fname is not None:
        # ext = fname.split('.')[-1]
        plt.savefig(fname, bbox_inches='tight')
    if show:
        plt.show()
   
    plt.close('all')

def plot_cactus_threshold(df, quantity, cumulative=False, title=None, show=False, fname=None, logscale=False, unit=None, labels=FakeLabels(), groupby=None):
    plt.figure(figsize=(10, 6))
    plt.gca().margins(0.02)

    df = df[df["plan-found"]==True]
    df = df[df['threshold'] != 'baseline']

    tmp = {}
    for solver, gdata in df.groupby(groupby):
        if solver not in CACTUS_EXCLUDE:
            tmp[str(solver[0])] = gdata[quantity].sort_values()
            # print("%s : %d" % (solver[0], len(tmp[solver[0]])))
            if cumulative:
                tmp[str(solver[0])] = tmp[str(solver[0])].cumsum()
    

    for solver, _ in sorted(tmp.items(), key=lambda x: (len(x[1]), x[0])):
        plot = plt.plot
        if logscale:
            plt.yscale('log')

        label = solver
        plot(range(0, len(tmp[label])),
             np.array(tmp[label]),
             LINES[label],
             color=COLORS[label],
             label=labels[label])

    plt.xlabel("Instances Solved", fontsize=FONTSIZE)

    lbl = quantity
    if unit is not None:
        lbl += "(%s)" % unit
    if cumulative:
        plt.ylabel("Cumulated Solving %s" % lbl, fontsize=FONTSIZE)
    else:
        plt.ylabel("Solving %s" % lbl, fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.legend(loc="lower right", frameon=False, prop={'size': FONTSIZE},
               labelspacing=LABELSPACING)
    if title is not None:
        plt.title(title, size=25, weight='bold')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, bbox_inches = 'tight')
    if show:
        plt.show()

    plt.close('all')




def main():
    """
    Aggregate data from the files res.csv of all problem sets, all runs and all problems, for each group
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)
    parser.add_argument('-o', '--output', type=str, default='./plots')
    parser.add_argument('-x', type=str, default = 'Expanded_states')
    parser.add_argument('-l', '--latex', action='store_true')
    parser.add_argument('-f', '--filter', action='store_true')
    parser.add_argument('-s', '--show', action='store_true')

    args, _ = parser.parse_known_args()

    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)

    pd.options.display.width = 0
    #df = pd.read_csv(os.path.join(args.i, 'planning_res.csv'))

    df = pd.read_csv(os.path.join(args.i))

    # ext = 'png'
    ext = 'pdf'
    labels = LABELS
    if args.latex:
        # with open('../paper/commands.tex', 'rt') as fh:
        #     plt.rcParams.update({
        #         "pgf.texsystem": "pdflatex",
        #         "pgf.preamble": fh.read(),
        #     })
        labels = LATEX_LABELS
        ext = 'pgf'

    df['_usage'] = df['parameters'].apply(extract_usage)
    df['time'] = df['wc-time'].apply(extract_time)

    if args.x == "time":
        logscale=True
        quantity = 'time'
        unit = 's'
    else:
        logscale=True
        quantity = 'Expanded states'
        unit=None


    valid_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    combinations = [
    ("majsp", "hadd"),
    ("kitting", "hadd"),
    ("matchcellar", "hff"),
    ("majsp", "hff"),
    ]

    for domain, heuristic in combinations:
        for usage in ['FA_minus', 'FA_plus', 'PA_minus', 'PA_plus']:
            # Filter the DataFrame for the current combination
            subset = df[
                (df['domain'] == domain) &
                (df['heuristic'] == heuristic) &
                (df['_usage'] == usage)
            ]

            if subset.empty:
                continue  # Skip if no data for this combo

            if domain == 'majsp':
                if heuristic == 'hff':
                    title = 'MAJSP - hff'
                else:
                    title = 'MAJSP - hadd'
            elif domain == 'kitting':
                title = 'Kitting - hadd'
            elif domain == 'matchcellar':   
                title = 'Matchcellar - hff'

            if domain == 'majsp':
                if heuristic == 'hff':
                    title = f'MAJSP - hff - {LABELS[usage]}'
                else:
                    title = f'MAJSP - hadd - {LABELS[usage]}'
            elif domain == 'kitting':
                title = f'Kitting - hadd - {LABELS[usage]}'
            elif domain == 'matchcellar':   
                title = f'Matchcellar - hff - {LABELS[usage]}'

            plot_cactus_threshold(subset, quantity, cumulative=False, logscale=logscale, fname=output + '/threshold/cactus-'+args.x+'_'+domain+'_'+heuristic+'_'+usage+'.' + ext, 
                        show=args.show, unit=unit, labels=labels, groupby=['threshold'], title=title)
    

    domains = ['majsp', 'kitting', 'matchcellar']
    best_planner = {
        "majsp" : [
            ("hadd", "FA_minus", 0.9),
            ("hadd", "FA_plus", 0.8),
            ("hadd", "PA_minus", 0.9),
            ("hadd", "PA_plus", 0.5),
            ("hff", "FA_minus", 0.9),
            ("hff", "FA_plus", 1),
            ("hff", "PA_minus", 0.9),
            ("hff", "PA_plus", 1)],
        "kitting" : [
            ("hadd", "FA_minus", 1),
            ("hadd", "FA_plus", 0.9),
            ("hadd", "PA_minus", 0.9),
            ("hadd", "PA_plus", 1)],
        "matchcellar" : [
            ("hff", "FA_minus", 0.5),
            ("hff", "FA_plus", 0.5),
            ("hff", "PA_minus", 0.5),
            ("hff", "PA_plus", 0.5),
            ("hff", "no_macros", 0)]
    }
    # Ensure threshold is numeric if needed
    df['threshold'] = pd.to_numeric(df['threshold'], errors='coerce')
    # Create a tuple column
    df['_filter_planner'] = list(zip(df['heuristic'], df['_usage'], df['threshold']))

    for domain in domains:
        subset = df[df['domain'] == domain]
        subset = subset[subset['_filter_planner'].isin(best_planner[domain])]

        # Extract the best solver 
        best_times = subset.groupby('instance')['time'].min().reset_index()
        best_rows = best_times.copy()
        best_rows['_usage'] = 'best_solver'
        best_rows['heuristic'] = ''
        best_rows['plan-found'] = best_rows['time'] < 600
        best_rows = best_rows.reindex(columns=df.columns)
        subset = pd.concat([subset, best_rows], ignore_index=True)

        groupby = ['heuristic', '_usage']
        # if domain == 'majsp':
        #     groupby = ['heuristic', '_usage']
        # else:
        #     groupby = ['_usage']
        if domain == 'majsp':
            title = 'MAJSP'
        else:
            title = domain.capitalize()

        plot_cactus(subset, quantity, cumulative=False, logscale=logscale, fname=output + '/best_planner/cactus-'+args.x+'_'+domain+'.' + ext, 
                    show=args.show, unit=unit, labels=labels, groupby=groupby, title=title)

    # coverage(df, True)


if __name__ == '__main__':
    main()
