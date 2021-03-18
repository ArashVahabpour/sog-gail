import argparse
import numpy as np
import pandas as pd
import yaml
from math import isnan
from itertools import chain


def group_to_range(group):
    group = ''.join(group.split())
    sign, g = ('-', group[1:]) if group.startswith('-') else ('', group)
    r = g.split('-', 1)
    r[0] = sign + r[0]
    r = sorted(int(__) for __ in r)
    return range(r[0], 1 + r[-1])


def range_expand(txt):
    """
    converts a string like "0-2,4-9,11" to a list of integers
    """
    ranges = chain.from_iterable(group_to_range(__) for __ in txt.split(','))
    return sorted(set(ranges))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--task',
    type=str,
    default='train',
    help='task to run: e.g. train, test, benchmark')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=4,
    help='number of random seeds to generate')
parser.add_argument(
    '--jobs',
    default=None,
    help='job rows in the spreadsheet, e.g. "0-2,4-9,11" . note that indices should begin from 0.')
args = parser.parse_args()

assert args.jobs, 'you must enter job indexes using --job-ids'
jobs = range_expand(args.jobs)
num_seeds = args.num_seeds if args.task == 'train' else 1

config = {"session_name": "run-all", "windows": []}
ex = pd.read_excel('jobs.xlsx')

for i in range(num_seeds):
    panes_list = []
    for job in jobs:
        job_args = ex.iloc[job, :]
        bool_keys = ['mujoco', 'vae-gail', 'infogail', 'sog-gail', 'adjust-scale', 'continuous']
        var_keys = ['name', 'env-name', 'infogail-coef', 'sog-gail-coef', 'latent-optimizer', 'block-size', 'save-dir', 'results-root', 'latent-dim', 'result-interval', 'save-interval', 'expert-filename', 'gpu-id']

        def template(key):
            value = job_args[key]
            try:
                if key not in var_keys + bool_keys or isnan(value):
                    return ''
            except:
                pass
            s = f'--{key}'
            if key in var_keys:
                try:
                    if int(value) == value:
                        value = int(value)
                except:
                    pass
                s += f' {value}'
            return s

        command = ' '.join([f'python {args.task}.py'] + list(filter(bool, map(template, ex.columns))))
        if args.task == 'train':
            command += f' --seed {i}'
        panes_list.append(command)

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)
