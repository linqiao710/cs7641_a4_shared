import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FormatStrFormatter, MaxNLocator
from os.path import basename

import config_util

cur_parma = "epsilonDecay"  # update for param
file_to_process = [
    {
        'name_pattern': '{}/*_{}_*_policies.csv',
        'index_name': 'iterations',
        'plot_cols': ['policies'],
        'plot_type': 'grid_map',
        'process': True
    },
    {
        'name_pattern': '{}/*_{}_*_values.csv',
        'index_name': 'iterations',
        'plot_cols': ['values'],
        'plot_type': 'grid_map',
        'process': True
    },
    {
        'name_pattern': '{}/*_{}_*_isVisited.csv',
        'index_name': 'iterations',
        'plot_cols': ['is_visited'],
        'index_name_combined': 'param_name',
        'plot_cols_combined': ['iterations', 'visited_states_number'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_{}_*_rewards.csv',
        'index_name': 'iterations',
        'plot_cols': ['time', 'reward'],
        'index_name_combined': 'param_name',
        'plot_cols_combined': ['iterations', 'time', 'rewards', 'delta'],
        'plot_type': 'line_mark',
        'process': True
    },
    {
        'name_pattern': '{}/*_{}_combined_multi_*.csv',
        'index_name': 'iterations',
        'plot_cols': [],  # iterate the columns in the file
        'plot_type': 'line_mark_multi',
        'process': True
    }
]


def plot_grid_map(column_name, title, grid_values, cur_env):
    plt.close()
    if grid_values.shape[1] > 12:
        fig = plt.figure(figsize=(20, 15))
    else:
        fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, grid_values.shape[1]), ylim=(0, grid_values.shape[0]))
    plt.title(title, fontsize=16)

    font_size_policy = 30
    font_size_value = 20

    for i in range(grid_values.shape[0]):
        for j in range(grid_values.shape[1]):
            y = grid_values.shape[0] - i - 1
            x = j
            p = plt.Rectangle([x, y], width=1, height=1, facecolor=cur_env.get_color(i, j), edgecolor='black',
                              linewidth=1.0)
            ax.add_patch(p)

            if column_name == 'policies':
                text = ax.text(x + 0.5, y + 0.5, cur_env.get_symbol(i, j, grid_values), weight='bold',
                               fontsize=font_size_policy,
                               horizontalalignment='center', verticalalignment='center', color='black')

            elif column_name == 'values' or column_name == 'is_visited':
                text = ax.text(x + 0.5, y + 0.5, cur_env.get_value(i, j, grid_values), fontsize=font_size_value,
                               horizontalalignment='center', verticalalignment='center', color='black')

    plt.axis('off')
    plt.xlim((0, grid_values.shape[1]))
    plt.ylim((0, grid_values.shape[0]))
    plt.tight_layout()

    return plt


def plot_line(df, column_name, legend_name, problem_path, title, x_label, y_label, clear_existing, mark, std):
    if clear_existing:
        plt.close()
        ax = plt.figure().gca()
        if df.index.name == 'iterations':
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title(title, fontsize=16)

        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.grid()
        plt.tight_layout()

    if std:
        cur_column_name = column_name + '_mean'
    else:
        cur_column_name = column_name

    if mark:
        plt.plot(df.index.values, df[cur_column_name], 'o-', linewidth=3, markersize=3,
                 label="{}_{}".format(legend_name, cur_column_name))
    else:
        plt.plot(df.index.values, df[cur_column_name], linewidth=3,
                 label="{}_{}".format(legend_name, cur_column_name))

    if std:
        plt.fill_between(df.index.values, df[cur_column_name] - df[column_name + "_std"],
                         df[cur_column_name] + df[column_name + "_std"], alpha=0.3)
    plt.legend(loc="best")
    return plt


def read_and_plot_map(cur_file_paths, output_dir, cur_file_to_process, cur_env_to_process, cur_prob_to_process,
                      plot_all=False):
    for cur_file_path in cur_file_paths:
        base_file_name = basename(cur_file_path)

        # it does not process multi here
        if 'multi' in base_file_name or 'combined' in base_file_name:
            continue

        output_file_name_regex = re.compile(
            '([A-Za-z]+)_([A-Za-z]+)_([0-9A-Za-z]+)_([A-Za-z]+)_([0-9A-Za-z]+[-][0-9A-Za-z]+|[0-9]+[.][0-9]+|[A-Za-z]+)_([A-Za-z]+)\.csv')
        cur_prob_path, cur_env_path, cur_env_size, param_name, param_val, file_name = output_file_name_regex.search(
            base_file_name).groups()

        sizes = cur_env_size.split('x')
        sizes = [int(i) for i in sizes]

        cur_index_key = 'index_name'
        cur_col_key = 'plot_cols'

        df = pd.read_csv(cur_file_path)
        df = df.set_index(cur_file_to_process[cur_index_key])
        if not plot_all:
            df = df.iloc[[0, -1]]

        for i, col_name in enumerate(cur_file_to_process[cur_col_key]):
            for index, row in df.iterrows():
                title = '{} [ {} ] \n{}{} [{}={}]'.format(cur_env_to_process['name'], cur_prob_to_process['name'],
                                                  'Iteration_#', index, param_name, param_val)

                grid_values = row[col_name].replace('[', '').replace(']', '').replace(',', '')
                grid_values = re.split('\s+', grid_values)
                grid_values = [float(i) for i in grid_values if len(i) > 0]
                grid_values = np.reshape(grid_values, (sizes[0], sizes[1]))

                # counts=np.array(np.unique(grid_values, return_counts=True)).T
                # print('{}={}: {}'.format(param_name, param_val, counts ))

                cur_env = cur_env_to_process['instance']
                p = plot_grid_map(column_name=col_name, title=title, grid_values=grid_values, cur_env=cur_env)

                out_file_name = '{}/{}_{}_{}_{}_{}_{}_{}.png'.format(output_dir, cur_prob_path, cur_env_path,
                                                                     cur_env_size,
                                                                     param_name, param_val, file_name, str(index))
                p.savefig(out_file_name)
                print("Plotting file {} to {}".format(cur_file_path, out_file_name))


def read_and_plot_line_multi(cur_file_paths, output_dir, cur_file_to_process, cur_env_to_process, cur_prob_to_process):
    for cur_file_path in cur_file_paths:
        base_file_name = basename(cur_file_path)

        output_file_name_regex = re.compile(
            '([A-Za-z]+)_([A-Za-z]+)_([0-9A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)_([A-Za-z]+)\.csv')
        cur_prob_path, cur_env_path, cur_env_size, param_name, param_val, file_name_1, file_name_2 = output_file_name_regex.search(
            base_file_name).groups()

        cur_index_key = 'index_name'

        df = pd.read_csv(cur_file_path).set_index(cur_file_to_process[cur_index_key])
        col_names = df.columns
        for i, col_name in enumerate(col_names):
            if i == 0:
                clear_existing = True

                y_label = file_name_2
                x_label = cur_file_to_process[cur_index_key]

                title = '{} [ {} ] \n{} vs {}'.format(cur_env_to_process['name'],
                                                      cur_prob_to_process['name'],
                                                      y_label, x_label)
            else:
                clear_existing = False

            p = plot_line(df=df, column_name=col_name, legend_name=param_name,
                          problem_path=cur_prob_to_process['path'],
                          title=title,
                          y_label=y_label, x_label=x_label, clear_existing=clear_existing, mark=True, std=False)

        out_file_name = '{}/{}_{}_{}_{}_{}_{}_{}.png'.format(output_dir, cur_prob_path, cur_env_path, cur_env_size,
                                                             param_name,
                                                             param_val, file_name_1, file_name_2)
        p.savefig(out_file_name)
        print("Plotting file {} to {}".format(cur_file_path, out_file_name))


def read_and_plot_line(cur_file_paths, output_dir, cur_file_to_process, cur_env_to_process, cur_prob_to_process):
    for cur_file_path in cur_file_paths:
        base_file_name = basename(cur_file_path)

        # it does not process multi here
        if 'multi' in base_file_name:
            continue

        output_file_name_regex = re.compile(
            '([A-Za-z]+)_([A-Za-z]+)_([0-9A-Za-z]+)_([A-Za-z]+)_([0-9A-Za-z]+[-][0-9A-Za-z]+|[0-9]+[.][0-9]+|[A-Za-z]+)_([A-Za-z]+)\.csv')
        cur_prob_path, cur_env_path, cur_env_size, param_name, param_val, file_name = output_file_name_regex.search(
            base_file_name).groups()

        if param_val == 'combined':
            cur_file_to_process['index_name_combined'] = param_name
            cur_index_key = 'index_name_combined'
            cur_col_key = 'plot_cols_combined'

        else:
            cur_index_key = 'index_name'
            cur_col_key = 'plot_cols'

        df = pd.read_csv(cur_file_path).set_index(cur_file_to_process[cur_index_key])

        for i, col_name in enumerate(cur_file_to_process[cur_col_key]):
            # it does not process if it does not contain this column
            if col_name not in df.columns:
                continue

            y_label = col_name
            x_label = cur_file_to_process[cur_index_key]

            title = '{} [ {} ] \n{} vs {}'.format(cur_env_to_process['name'],
                                                  cur_prob_to_process['name'],
                                                  y_label, x_label)

            p = plot_line(df=df, column_name=col_name, legend_name=cur_prob_to_process['path'],
                          problem_path=cur_prob_to_process['path'],
                          title=title,
                          y_label=y_label,
                          x_label=x_label, clear_existing=True, mark=True, std=False)

            out_file_name = '{}/{}_{}_{}_{}_{}_{}.png'.format(output_dir, cur_prob_path, cur_env_path, cur_env_size,
                                                              param_name, param_val, y_label)
            p.savefig(out_file_name)
            print("Plotting file {} to {}".format(cur_file_path, out_file_name))


def read_and_plot_problem(cur_env_to_process, cur_prob_to_process):
    problem_input_path = '{}/{}/{}/{}/'.format('output', cur_env_to_process['path'], cur_env_to_process['size'],
                                               cur_prob_to_process['path'])
    problem_output_path = '{}/{}/{}/{}/'.format('image', cur_env_to_process['path'], cur_env_to_process['size'],
                                                cur_prob_to_process['path'])

    if not os.path.exists(problem_output_path):
        os.makedirs(problem_output_path)

    for cur_file_to_process in file_to_process:
        file_paths = glob.glob(cur_file_to_process['name_pattern'].format(problem_input_path, cur_parma))

        if len(file_paths) == 0 or not cur_file_to_process['process']:
            continue

        if cur_file_to_process['plot_type'] == 'line_mark':
            read_and_plot_line(cur_file_paths=file_paths, output_dir=problem_output_path,
                               cur_file_to_process=cur_file_to_process, cur_env_to_process=cur_env_to_process,
                               cur_prob_to_process=cur_prob_to_process)

        elif cur_file_to_process['plot_type'] == 'line_mark_multi':
            read_and_plot_line_multi(cur_file_paths=file_paths, output_dir=problem_output_path,
                                     cur_file_to_process=cur_file_to_process, cur_env_to_process=cur_env_to_process,
                                     cur_prob_to_process=cur_prob_to_process)

        elif cur_file_to_process['plot_type'] == 'grid_map':
            read_and_plot_map(cur_file_paths=file_paths, output_dir=problem_output_path,
                              cur_file_to_process=cur_file_to_process, cur_env_to_process=cur_env_to_process,
                              cur_prob_to_process=cur_prob_to_process)


if __name__ == '__main__':
    for env_to_process in config_util.envs_to_process:
        if not env_to_process['process']:
            continue

        for prob_to_process in config_util.problem_to_process:
            if not prob_to_process['process']:
                continue

            print("================================")
            print("Env_to_process:{}, Problem_to_process:{}".format(env_to_process['name'], prob_to_process['name']))
            read_and_plot_problem(env_to_process, prob_to_process)
