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
    help='task to run: e.g. train, test')
parser.add_argument(
    '--test-task',
    type=str,
    default=None,
    help='test task: e.g. benchmark, plot, play')
parser.add_argument(
    '--num-seeds',
    type=int,
    default=1,
    help='number of random seeds to generate')
parser.add_argument(
    '--jobs',
    default=None,
    help='job rows in the spreadsheet, e.g. "0-2,4-9,11" . note that indices should begin from 0.')
args = parser.parse_args()

assert args.jobs, 'you must enter job indexes using --jobs'
jobs = range_expand(args.jobs)
jobs = [job - 2 for job in jobs]
num_seeds = args.num_seeds if args.task == 'train' else 1

config = {"session_name": "run-all", "windows": []}
ex = pd.read_excel('jobs.xlsx')

for i in range(num_seeds):
    panes_list = []
    for job in jobs:
        job_args = ex.iloc[job, :]
        bool_keys = ['mujoco', 'vae-gail', 'vae-cheat', 'infogail', 'sog-gail', 'adjust-scale', 'continuous', 'shared']
        var_keys = ['name', 'env-name', 'infogail-coef', 'sog-gail-coef', 'latent-optimizer', 'block-size', 'save-dir', 'results-root', 'latent-dim', 'result-interval', 'save-interval', 'gail-experts-dir', 'expert-filename', 'gpu-id', 'num-clusters',  'radii']

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
                if key == 'name' and i > 0:
                    s += f'.seed{i}'
            return s

        assert args.task in ['train', 'test']
        command = ' '.join([f'python {args.task}.py'] + list(filter(bool, map(template, ex.columns))))
        if args.task == 'train':
            command += f' --seed {i}'
        elif args.task == 'test' and args.test_task is not None:
            command += f' --test-task {args.test_task}'
        panes_list.append(command)

    config["windows"].append({
        "window_name": "seed-{}".format(i),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), width=1000, default_flow_style=False)
