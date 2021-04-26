import array
import copy
import os
import random
from itertools import product

import numpy as np
import pandas as pd

import config_util
import numpy as np
import hiive.mdptoolbox
import hiive.mdptoolbox.mdp as mdp


def convert_P_R(cur_env):
    flstates = cur_env.nS
    flactions = cur_env.nA
    fltrans = np.zeros((flactions, flstates, flstates))
    flrewards = np.zeros((flstates, flactions))
    for state in cur_env.P:
        for action in cur_env.P[state]:
            for opt in cur_env.P[state][action]:
                fltrans[action][state][opt[1]] += opt[0]
                flrewards[state][action] += opt[2]
    return fltrans, flrewards


def ql_experiments(cur_P, cur_R, cur_env_to_process, cur_prob_to_process, cur_problem_output_path):
    learning_rate_list = [0.1, 0.3, 0.5, 0.9]
    epsilon_list = [0.3, 0.5, 0.7, 0.9]
    epsilon_decay_list = [0.000005, 0.00005, 0.0005, 0.005]
    gamma_list = [0.2, 0.4, 0.8, 0.99]
    rewards_col_names = ['iterations', 'time', 'rewards', 'delta']
    values_col_names = ['iterations', 'values']
    policies_col_names = ['iterations', 'policies']
    q_tables_col_names = ['iterations', 'q_table']
    is_visited_col_names = ['iterations', 'is_visited']
    param_name_list = ['learningRate', 'epsilon', 'epsilonDecay', 'gamma']
    param_name = param_name_list[2]  # update for param
    param_rewards_col_names = [param_name, 'iterations', 'time', 'rewards', 'delta']
    param_multi_col_names = [str(g) for g in epsilon_decay_list]  # update for param

    param_rewards = []
    param_multi_time_df = pd.DataFrame()
    param_multi_rewards_df = pd.DataFrame()
    param_multi_delta_df = pd.DataFrame()

    for learning_rate, epsilon, epsilon_decay, gamma in product([0.9], [0.3], epsilon_decay_list,
                                                                [0.95]):  # update for param
        p_iter = copy.deepcopy(cur_P)
        r_iter = copy.deepcopy(cur_R)
        param_val = epsilon_decay  # update for param
        iter_num = 8e5
        stat_frequency = 8e3
        ql = mdp.QLearning(transitions=p_iter, reward=r_iter, gamma=gamma, alpha=learning_rate, epsilon=epsilon,
                           epsilon_decay=epsilon_decay, n_iter=iter_num,
                           run_stat_frequency=stat_frequency)
        ql.verbose = True
        ql.run()
        stats = ql.run_stats
        mdp_prob = ql

        # ===================#===================#===================
        param_reward = [param_val]
        param_stats = stats[-1]['Iteration'], stats[-1]['Time'], stats[-1]['Reward'], stats[-1]['Error']
        param_reward.extend(param_stats)
        param_rewards.append(param_reward)
        # ===================#===================#===================
        rewards = []
        values = []
        policies = []
        for item in stats:
            rewards.append([item['Iteration'], item['Time'], item['Reward'], item['Error']])

        policies.append([stats[-1]['Iteration'], list(mdp_prob.policy)])
        values.append([stats[-1]['Iteration'], list(mdp_prob.V)])
        # ===================#===================#===================
        if env_to_process['type'] == 'non-grid':
            policyarr = np.array(list(mdp_prob.policy)).reshape(mdp_prob.S).astype(str)
            valuearr = np.round(np.array(list(mdp_prob.V)).reshape(mdp_prob.S), 2)
            policyarr[policyarr == '0'] = 'W'
            policyarr[policyarr == '1'] = 'C'
            policyviz = (
                np.asarray([a + " " + str(v) for a, v in zip(policyarr.flatten(), valuearr.flatten())])).reshape(
                mdp_prob.S)

            print("")
            print("Policy Iteration: Optimal Policy")
            line = ""
            for val in policyviz:
                line += (" {} >".format(val))

            f = open(cur_problem_output_path + "optimal_{}_{}.txt".format(param_name, param_val), "w")
            f.write(line)
            f.close()
        # ===================#===================#===================
        rewards_df = pd.DataFrame(rewards, columns=rewards_col_names).set_index(rewards_col_names[0])
        rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
                                                                                     cur_env_to_process['path'],
                                                                                     cur_env_to_process['size'],
                                                                                     param_name,
                                                                                     param_val,
                                                                                     'rewards')
        rewards_df.to_csv(rewards_df_name)
        print("Creating {} ".format(rewards_df_name))
        # ===================#===================#===================
        values_df = pd.DataFrame(values, columns=values_col_names).set_index(values_col_names[0])
        values_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
                                                                                    cur_env_to_process['path'],
                                                                                    cur_env_to_process['size'],
                                                                                    param_name,
                                                                                    param_val,
                                                                                    'values')
        values_df.to_csv(values_df_name)
        print("Creating {} ".format(values_df_name))
        # ===================#===================#===================
        policies_df = pd.DataFrame(policies, columns=policies_col_names).set_index(policies_col_names[0])
        policies_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
                                                                                      cur_env_to_process['path'],
                                                                                      cur_env_to_process['size'],
                                                                                      param_name,
                                                                                      param_val,
                                                                                      'policies')
        policies_df.to_csv(policies_df_name)
        print("Creating {} ".format(policies_df_name))
        # ===================#===================#===================
        # q_tables_df = pd.DataFrame(q_tables, columns=q_tables_col_names).set_index(q_tables_col_names[0])
        # q_tables_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
        #                                                                               cur_env_to_process['path'],
        #                                                                               cur_env_to_process['size'],
        #                                                                               param_name,
        #                                                                               param_val,
        #                                                                               'qtables')
        # q_tables_df.to_csv(q_tables_df_name)
        # print("Creating {} ".format(q_tables_df_name))
        # ===================#===================#===================
        # is_visited_df = pd.DataFrame(is_visited, columns=is_visited_col_names).set_index(q_tables_col_names[0])
        # is_visited_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
        #                                                                                 cur_env_to_process['path'],
        #                                                                                 cur_env_to_process['size'],
        #                                                                                 param_name,
        #                                                                                 param_val,
        #                                                                                 'isVisited')
        # is_visited_df.to_csv(is_visited_df_name)
        # print("Creating {} ".format(is_visited_df_name))
        # ===================#===================#===================
        param_multi_time_df = pd.concat([param_multi_time_df, rewards_df['time']], ignore_index=True, axis=1)

        # ===================#===================#===================
        param_multi_rewards_df = pd.concat([param_multi_rewards_df, rewards_df['rewards']], ignore_index=True,
                                           axis=1)
        # ===================#===================#===================
        param_multi_delta_df = pd.concat([param_multi_delta_df, rewards_df['delta']], ignore_index=True,
                                         axis=1)
        # ===================#===================#===================

    param_rewards_df = pd.DataFrame(param_rewards, columns=param_rewards_col_names).set_index(
        param_rewards_col_names[0])
    param_rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
                                                                                       cur_env_to_process['path'],
                                                                                       cur_env_to_process['size'],
                                                                                       param_name,
                                                                                       'combined',
                                                                                       'rewards')
    param_rewards_df.to_csv(param_rewards_df_name)
    print("Creating {} ".format(param_rewards_df_name))
    # ===================#===================#===================

    param_multi_time_df = param_multi_time_df.fillna(method='ffill')
    param_multi_time_df.columns = param_multi_col_names
    s = pd.Series(range(1, len(param_multi_time_df) + 1))
    param_multi_time_df.set_index(s)
    param_multi_time_df.index.name = 'iterations'

    param_multi_time_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
        cur_prob_to_process['path'],
        cur_env_to_process['path'],
        cur_env_to_process['size'],
        param_name, 'combined', 'multi', 'time')
    param_multi_time_df.to_csv(param_multi_time_df_name)
    print("Creating {} ".format(param_multi_time_df_name))
    # ===================#===================#===================

    param_multi_rewards_df = param_multi_rewards_df.fillna(method='ffill')
    param_multi_rewards_df.columns = param_multi_col_names
    s = pd.Series(range(1, len(param_multi_rewards_df) + 1))
    param_multi_rewards_df.set_index(s)
    param_multi_rewards_df.index.name = 'iterations'

    param_multi_rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
        cur_prob_to_process['path'],
        cur_env_to_process['path'],
        cur_env_to_process['size'],
        param_name, 'combined', 'multi', 'rewards')
    param_multi_rewards_df.to_csv(param_multi_rewards_df_name)
    print("Creating {} ".format(param_multi_rewards_df_name))
    # ===================#===================#===================
    param_multi_delta_df = param_multi_delta_df.fillna(method='ffill')
    param_multi_delta_df.columns = param_multi_col_names
    s = pd.Series(range(1, len(param_multi_delta_df) + 1))
    param_multi_delta_df.set_index(s)
    param_multi_delta_df.index.name = 'iterations'

    param_multi_delta_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
        cur_prob_to_process['path'],
        cur_env_to_process['path'],
        cur_env_to_process['size'],
        param_name, 'combined', 'multi', 'delta')
    param_multi_delta_df.to_csv(param_multi_delta_df_name)
    print("Creating {} ".format(param_multi_delta_df_name))
    # ===================#===================#===================


# ==============#=====================
# ==============#=====================
# ==============#=====================
def pi_vi_experiments(cur_P, cur_R, cur_env_to_process, cur_prob_to_process, cur_problem_output_path):
    eps_list = [0.0001, 0.0003, 0.0005, 0.0007]
    gamma_list = [0.2, 0.4, 0.8, 0.99]

    rewards_col_names = ['iterations', 'time', 'rewards', 'delta']
    values_col_names = ['iterations', 'values']
    policies_col_names = ['iterations', 'policies']
    param_name_list = ['epsilon', 'gamma']
    param_name = param_name_list[1]  # update for param
    param_rewards_col_names = [param_name, 'iterations', 'time', 'rewards', 'delta']
    param_multi_col_names = [str(g) for g in gamma_list]  # update for param

    param_rewards = []
    param_multi_time_df = pd.DataFrame()
    param_multi_rewards_df = pd.DataFrame()
    param_multi_delta_df = pd.DataFrame()

    for epsilon, gamma in product([0.0001], [0.99]):  # update for param
        p_iter = copy.deepcopy(cur_P)
        r_iter = copy.deepcopy(cur_R)
        param_val = gamma  # update for param
        if cur_prob_to_process['path'] == 'pi':
            pi = mdp.PolicyIteration(transitions=p_iter, reward=r_iter, gamma=gamma, max_iter=3000, run_stat_frequency=1)
            pi.run()
            stats = pi.run_stats
            mdp_prob = pi
        elif cur_prob_to_process['path'] == 'vi':
            vi = mdp.ValueIteration(transitions=p_iter, reward=r_iter, gamma=gamma, max_iter=3000, run_stat_frequency=1)
            vi.run()
            stats = vi.run_stats
            mdp_prob = vi

        # ===================#===================#===================
        param_reward = [param_val]
        param_stats = stats[-1]['Iteration'], stats[-1]['Time'], stats[-1]['Reward'], stats[-1]['Error']
        param_reward.extend(param_stats)
        param_rewards.append(param_reward)
        # ===================#===================#===================
        rewards = []
        values = []
        policies = []
        for item in stats:
            rewards.append([item['Iteration'], item['Time'], item['Reward'], item['Error']])

        policies.append([stats[-1]['Iteration'], list(mdp_prob.policy)])
        values.append([stats[-1]['Iteration'], list(mdp_prob.V)])
        # ===================#===================#===================
        if env_to_process['type'] == 'non-grid':
            policyarr = np.array(list(mdp_prob.policy)).reshape(mdp_prob.S).astype(str)
            valuearr = np.round(np.array(list(mdp_prob.V)).reshape(mdp_prob.S), 2)
            policyarr[policyarr == '0'] = 'W'
            policyarr[policyarr == '1'] = 'C'
            policyviz = (np.asarray([a + " " + str(v) for a, v in zip(policyarr.flatten(), valuearr.flatten())])).reshape(
                mdp_prob.S)

            print("")
            print("Policy Iteration: Optimal Policy")
            line = ""
            for val in policyviz:
                line += (" {} >".format(val))

            f = open(cur_problem_output_path + "optimal_{}_{}.txt".format(param_name, param_val), "w")
            f.write(line)
            f.close()
    #     # ===================#===================#===================
    #     rewards_df = pd.DataFrame(rewards, columns=rewards_col_names).set_index(rewards_col_names[0])
    #     rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
    #                                                                                  cur_env_to_process['path'],
    #                                                                                  cur_env_to_process['size'],
    #                                                                                  param_name,
    #                                                                                  param_val,
    #                                                                                  'rewards')
    #     rewards_df.to_csv(rewards_df_name)
    #     print("Creating {} ".format(rewards_df_name))
    #     # ===================#===================#===================
    #     values_df = pd.DataFrame(values, columns=values_col_names).set_index(values_col_names[0])
    #     values_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
    #                                                                                 cur_env_to_process['path'],
    #                                                                                 cur_env_to_process['size'],
    #                                                                                 param_name,
    #                                                                                 param_val,
    #                                                                                 'values')
    #     values_df.to_csv(values_df_name)
    #     print("Creating {} ".format(values_df_name))
    #     # ===================#===================#===================
    #     policies_df = pd.DataFrame(policies, columns=policies_col_names).set_index(policies_col_names[0])
    #     policies_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
    #                                                                                   cur_env_to_process['path'],
    #                                                                                   cur_env_to_process['size'],
    #                                                                                   param_name,
    #                                                                                   param_val,
    #                                                                                   'policies')
    #     policies_df.to_csv(policies_df_name)
    #     print("Creating {} ".format(policies_df_name))
    #     # ===================#===================#===================
    #     param_multi_time_df = pd.concat([param_multi_time_df, rewards_df['time']], ignore_index=True, axis=1)
    #
    #     # ===================#===================#===================
    #     param_multi_rewards_df = pd.concat([param_multi_rewards_df, rewards_df['rewards']], ignore_index=True,
    #                                        axis=1)
    #
    #     # ===================#===================#===================
    #     param_multi_delta_df = pd.concat([param_multi_delta_df, rewards_df['delta']], ignore_index=True,
    #                                      axis=1)
    #     # ===================#===================#===================
    #
    # param_rewards_df = pd.DataFrame(param_rewards, columns=param_rewards_col_names).set_index(
    #     param_rewards_col_names[0])
    # param_rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}.csv').format(cur_prob_to_process['path'],
    #                                                                                    cur_env_to_process['path'],
    #                                                                                    cur_env_to_process['size'],
    #                                                                                    param_name,
    #                                                                                    'combined',
    #                                                                                    'rewards')
    # param_rewards_df.to_csv(param_rewards_df_name)
    # print("Creating {} ".format(param_rewards_df_name))
    # # ===================#===================#===================
    #
    # param_multi_time_df = param_multi_time_df.fillna(method='ffill')
    # param_multi_time_df.columns = param_multi_col_names
    # s = pd.Series(range(1, len(param_multi_time_df) + 1))
    # param_multi_time_df.set_index(s)
    # param_multi_time_df.index.name = 'iterations'
    #
    # param_multi_time_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
    #     cur_prob_to_process['path'],
    #     cur_env_to_process['path'],
    #     cur_env_to_process['size'],
    #     param_name, 'combined', 'multi', 'time')
    # param_multi_time_df.to_csv(param_multi_time_df_name)
    # print("Creating {} ".format(param_multi_time_df_name))
    # # ===================#===================#===================
    #
    # param_multi_rewards_df = param_multi_rewards_df.fillna(method='ffill')
    # param_multi_rewards_df.columns = param_multi_col_names
    # s = pd.Series(range(1, len(param_multi_rewards_df) + 1))
    # param_multi_rewards_df.set_index(s)
    # param_multi_rewards_df.index.name = 'iterations'
    #
    # param_multi_rewards_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
    #     cur_prob_to_process['path'],
    #     cur_env_to_process['path'],
    #     cur_env_to_process['size'],
    #     param_name, 'combined', 'multi', 'rewards')
    # param_multi_rewards_df.to_csv(param_multi_rewards_df_name)
    # print("Creating {} ".format(param_multi_rewards_df_name))
    # # ===================#===================#===================
    # param_multi_delta_df = param_multi_delta_df.fillna(method='ffill')
    # param_multi_delta_df.columns = param_multi_col_names
    # s = pd.Series(range(1, len(param_multi_delta_df) + 1))
    # param_multi_delta_df.set_index(s)
    # param_multi_delta_df.index.name = 'iterations'
    #
    # param_multi_delta_df_name = (cur_problem_output_path + '{}_{}_{}_{}_{}_{}_{}.csv').format(
    #     cur_prob_to_process['path'],
    #     cur_env_to_process['path'],
    #     cur_env_to_process['size'],
    #     param_name, 'combined', 'multi', 'delta')
    # param_multi_delta_df.to_csv(param_multi_delta_df_name)
    # print("Creating {} ".format(param_multi_delta_df_name))


if __name__ == "__main__":
    seed = config_util.random_state
    random.seed(seed)
    np.random.seed(seed)

    for env_to_process in config_util.envs_to_process:
        if not env_to_process['process']:
            continue

        if env_to_process['type'] == 'grid':
            env = env_to_process['instance']
            P, R = convert_P_R(env)
        elif env_to_process['type'] == 'non-grid':
            P, R = env_to_process['instance']

        for prob_to_process in config_util.problem_to_process:
            if not prob_to_process['process']:
                continue

            print("================================")
            print("Env_to_process:{}, Problem_to_process:{}".format(env_to_process['name'],
                                                                    prob_to_process['name']))

            problem_output_path = '{}/{}/{}/{}/'.format('output', env_to_process['path'], env_to_process['size'],
                                                        prob_to_process['path'])
            if not os.path.exists(problem_output_path):
                os.makedirs(problem_output_path)

            if prob_to_process['path'] == 'pi' or prob_to_process['path'] == 'vi':
                pi_vi_experiments(P, R, env_to_process, prob_to_process, problem_output_path)
            elif prob_to_process['path'] == 'ql':
                ql_experiments(P, R, env_to_process, prob_to_process, problem_output_path)
