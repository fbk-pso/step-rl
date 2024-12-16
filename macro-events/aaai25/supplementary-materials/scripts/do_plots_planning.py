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

from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np
from tabulate import tabulate


LINES = {'no_macros'    : '-',
         'FA_minus'     : '--',
         'FA_plus'      : '-.',
         'PA_minus'     : '-',
         'PA_plus'      : '--'}

COLORS = {'no_macros'   : 'k',
          'FA_minus'    : 'y',
          'FA_plus'     : 'b',
          'PA_minus'    : 'g',
          'PA_plus'     : 'c',
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
          'optic'       : 'b'}

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
          'FA_minus' : 'FA-',
          'FA_plus' : 'FA+',
          'PA_minus' : 'PA-',
          'PA_plus' : 'PA+',}

LATEX_LABELS = {'painter'     : '\\painter',
                'majsp'       : '\\majsp',
                'tms'         : '\\TMS',
                'satellite'   : '\\Satellite',
                'mapanalyser' : '\\MAP',
                'driverlog'   : '\\Driverlog',
                'matchcellar' : '\\Matchcellar',
                'floortile'   : '\\Floortile',
                'tamer'       : '\\tamer',
                'optic'       : '\\optic\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ',
                'mono'        : '\\monostn',
                'delta'       : '\\deltastn',
                'clone'       : '\\clonestn',
                'lp'          : '\\linprog',
                'smtmono'     : '\\smt',
                'smtinc'      : '\\smtinc',
                'smtsua'      : '\\smtsua'}


CACTUS_EXCLUDE = ['clone']

FONTSIZE=16

LABELSPACING=0.25


class FakeLabels:
    def __getitem__(self, x):
        return x

def extract_usage(x):
    if x.endswith('hadd') or x.endswith('hff'):
        return 'no_macros'
    elif x.endswith('FA_minus'):
        return 'FA_minus'
    elif x.endswith('PA_minus'):
        return 'PA_minus'
    elif x.endswith('PA_plus'):
        return 'PA_plus'
    elif x.endswith('FA_plus'):
        return 'FA_plus'

def extract_time(x):
    if x < 1:
        return 1
    elif x > 600:
        return 600
    else:
        return x


def plot_cactus(df, quantity, cumulative=False, title=None, show=False, fname=None, logscale=False, unit=None, labels=FakeLabels()):
    plt.figure(figsize=(10, 6))
    plt.gca().margins(0.02)

    df = df[~df["solved"].isna()]

    tmp = {}
    for solver, gdata in df.groupby(['_usage']):
        if solver not in CACTUS_EXCLUDE:
            tmp[solver[0]] = gdata[quantity].sort_values()
            print("%s : %d" % (solver[0], len(tmp[solver[0]])))
            if cumulative:
                tmp[solver[0]] = tmp[solver[0]].cumsum()

    for solver, _ in sorted(tmp.items(), key=lambda x: (len(x[1]), ord(x[0][0]))):
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

    plt.legend(loc="upper left", frameon=False, prop={'size': FONTSIZE},
               labelspacing=LABELSPACING)
    if title is not None:
        plt.title(title, size='large', weight='bold')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, bbox_inches = 'tight')
    if show:
        plt.show()

    plt.close('all')


pd.set_option('display.max_rows', 500)
TODO = {('lp', 'delta'), ('smtmono', 'delta'), ('mono', 'delta')}
def plot_scatters(objective, df, show=False, fname=None, logscale=False, groupby=None, labels=FakeLabels(), showlegend=True):

    ito = df['real'] > 600
    imo = df['MemoryError'] == True

    if objective == 'time':
        to = 610
        mo = 625
        lim = 630
        inf = 3

        if logscale:
            to = 700
            mo = 800
            lim = 880

        df.loc[ito, '_time'] = to
        df.loc[imo, '_time'] = mo

    else:

        df['_Expanded_states'] = df['Expanded states']
        to = max(df['_Expanded_states'].max(), df['_Expanded_states'].max())  + 5
        mo = to + 10
        lim = to + 15
        inf = 1

        if logscale:
            m = max(df['_Expanded_states'].max(), df['_Expanded_states'].max())
            to = m + m* 3/10
            mo = to + to* 3/10
            lim = to + to* 3.3/5


        df.loc[ito, '_Expanded_states'] = to
        df.loc[imo, '_Expanded_states'] = mo

    #df.to_csv('plots/prova.csv')


    for solver1, gdata1 in df.groupby(['_usage']):
        label1 = solver1
        gdata1 = gdata1.set_index('instance')
        for solver2, gdata2 in df.groupby(['_usage']):
            label2 = solver2
            gdata2 = gdata2.set_index('instance')
            #if (solver1, solver2) not in TODO: continue

            # to = max(df[df['_usage']==label1[0]]['_Expanded_states'].max(), df[df['_usage']==label2[0]]['_Expanded_states'].max())  + 5
            # mo = to + 10
            # lim = to + 15
            # inf = 1

            # if logscale:
            #     m = max(df[df['_usage']==label1[0]]['_Expanded_states'].max(), df[df['_usage']==label2[0]]['_Expanded_states'].max())
            #     to = m + m* 3/10
            #     mo = to + to* 3/10
            #     lim = to + to* 3.3/5


            # df.loc[ito, '_Expanded_states'] = to
            # df.loc[imo, '_Expanded_states'] = mo


            plt.figure(figsize=(8.5, 6))
            plt.gca().margins(0.02)
            if logscale:
                plt.yscale('log')
                plt.xscale('log')
            plt.xlim(inf, lim)
            plt.ylim(inf, lim)

            plt.plot([inf, mo], [inf, mo], '--', color='k')
            plt.plot([to, to], [inf, to], '--', color='b')
            plt.plot([mo, mo], [inf, mo], '--', color='g')
            plt.plot([inf, to], [to, to], '--', color='b')
            plt.plot([inf, mo], [mo, mo], '--', color='g')
            plt.gcf().text(-0.05, 0.96, 'TO', {'color':'b'}, transform=plt.gca().transAxes)
            plt.gcf().text(-0.05, 0.99, 'MO', {'color':'g'}, transform=plt.gca().transAxes)

            j = gdata1.join(gdata2, on='instance', rsuffix='_s2', lsuffix='_s1')
            if groupby is None:
                if objective == "time":
                    plt.scatter(j['_time_s1'], j['_time_s2'], color='r', marker='P', s=15)
                else:
                    plt.scatter(j['_Expanded_states_s1'], j['_Expanded_states_s2'], color='r', marker='P', s=15)
            else:
                for x, _ in df.groupby([groupby]):
                    d = j[j[groupby + '_s1'] == x]
                    plt.scatter(d['_time_s1'], d['_time_s2'], color=COLORS[x], marker=MARKERS[x], label=labels[x], s=30)

            plt.xlabel(labels[label1[0]], fontsize=FONTSIZE)
            plt.ylabel(labels[label2[0]], fontsize=FONTSIZE)

            plt.gca().set_aspect('equal', 'box')

            if groupby and showlegend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                           borderaxespad=0., prop={'size':12})

            plt.tight_layout()

            if show:
                plt.show()

            if fname is not None:
                plt.savefig(fname % (label1[0], label2[0]), bbox_inches = 'tight')

            plt.close('all')


def pr(x):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(x)


def print_table(table, title, latex):
    print('-' *80)
    print(title)
    print('-' *80)
    if latex:
        table = table.rename(columns=LATEX_LABELS, index=LATEX_LABELS)
        print(table.to_latex(escape=False))
    else:
        print(tabulate(table, headers='keys'))
    print('-' * 80)
    print('\n\n')


def pivot_sorter(y):
    k = 'A'
    if 'Total' in y:
        k = 'D'
    return k + y

COLUMN_ORDER = {
    'no_macros' : 1,
    'FA_minus' : 2,
    'FA_plus' : 3,
    'PA_minus' : 4,
    'PA_plus' : 5,
}

def coverage(df, latex, filter=None):
    df["_countsolved"] = (df["solved"].apply(lambda x: 1 if x == True else 0))

    df = df.set_index('instance')

    j = df.join(df[df['_usage'] == "no_macros"], on='instance', rsuffix='_nomacros', lsuffix='')
    j["_best_than_no_macros"] = ((j["real"] < j["real_nomacros"]) & (j["solved"] == True)).apply(lambda x: 1 if x == True else 0)

    table = pd.pivot_table(j, values=['_countsolved', '_best_than_no_macros'], index='set', columns='_usage', aggfunc={'_countsolved':'sum', '_best_than_no_macros':'sum'}, fill_value=0, margins=True, margins_name='Total')
    table = table.swaplevel(1,0,axis=1).sort_index(axis=1)
    table.drop(columns=['Total'], inplace=True)
    table = table.stack(level=1).groupby(level=[0]).agg(lambda x: f"{x[1]} ({x[0]})")
    table.sort_index(inplace=True, key=lambda idx: idx.map(pivot_sorter))
    cols = list(table.columns.values)
    cols = [x for x in cols if not filter or filter(x)]
    cols.sort(key=lambda x: COLUMN_ORDER.get(x, 0))

    table = table[cols]

    # anyplan = df[~(df['any_plan'].isnull())]
    # tadd= pandas.pivot_table(anyplan, values='any_plan', index='domain_id', columns='planner', aggfunc=len, fill_value=0, margins=True, margins_name='Total')
    # tadd.drop(columns=['Total'], inplace=True)
    # tadd.sort_index(inplace=True, key=lambda idx: idx.map(pivot_sorter))
    # cols = list(tadd.columns.values)
    # cols = [x for x in cols if not filter or filter(x)]
    # cols.sort(key=lambda x: COLUMN_ORDER.get(x, 0))
    # tadd = tadd[cols]

    # for c in cols:
    #     table[c] = table[c].astype(str) + '(' + tadd[c].astype(str) + ')'

    print_table(table, 'Coverage Results', latex)


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
        os.mkdir(output)

    pd.options.display.width = 0
    #df = pd.read_csv(os.path.join(args.i, 'planning_res.csv'))

    df = pd.read_csv(os.path.join(args.i))

    ext = 'png'
    labels = LABELS
    if args.latex:
        # with open('../paper/commands.tex', 'rt') as fh:
        #     plt.rcParams.update({
        #         "pgf.texsystem": "pdflatex",
        #         "pgf.preamble": fh.read(),
        #     })
        labels = LATEX_LABELS
        ext = 'pgf'

    df['_usage'] = df['group'].apply(extract_usage)
    df['_time'] = df['real'].apply(extract_time)

    if args.x == "time":
        logscale=True
        quantity = '_time'
        unit = 's'
    else:
        logscale=True
        quantity = 'Expanded states'
        unit=None


    plot_scatters(args.x, df, fname=output + "/scatter-"+ args.x +"-%s-%s." + ext, groupby=None, logscale=logscale,  labels=labels, showlegend=False)

    plot_cactus(df, quantity, cumulative=False, logscale=logscale,
                fname=output + '/cactus-'+args.x+'.' + ext, show=args.show,
                unit=unit, labels=labels)

    coverage(df, True)



if __name__ == '__main__':
    main()


