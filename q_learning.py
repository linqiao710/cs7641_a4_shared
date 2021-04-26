import random
import time
from time import clock
import numpy as np

import config_util


def convert_q_2_policy(q_table):
    policy = np.zeros(len(q_table))
    for s in range(len(q_table)):
        policy[s] = greedy(q_table, s)
    return policy


def convert_q_2_value(q_table):
    value = np.zeros(len(q_table))
    for s in range(len(q_table)):
        value[s] = np.max(q_table[s])
    return value


def eps_greedy(Q, s, eps):
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the greedy action
        return greedy(Q, s)


# The greedy policy is implemented by returning the index that corresponds to the maximum Q value in state s
def greedy(Q, s):
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes, to_print=False):
    tot_rew = []
    state = env.reset()
    for _ in range(num_episodes):
        done = False
        game_rew = 0
        while not done:
            next_state, rew, done, _ = env.step(greedy(Q, state))
            env.render()
            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)
    if to_print:
        print('Mean score: %.3f of %i games!' % (np.mean(tot_rew), num_episodes))
    else:
        return np.mean(tot_rew)


def get_reward(env, Q, num_games, max_step=800):
    state = env.reset()
    tot_rew = 0
    for _ in range(num_games):
        done = False
        step = 0
        while not done and step <= max_step:
            next_state, reward, done, _ = env.step(greedy(Q, state))
            state = next_state
            tot_rew += reward
            step += 1
            if done:
                state = env.reset()

    return tot_rew


def Q_learning(env, lr=0.01, num_episodes=5000000, num_games=100, eps=0.3, eps_decay=0.00005, gamma=0.95,
               threshold_val=0.0000001,
               threshold_it=00000):
    seed = config_util.random_state
    random.seed(seed)
    np.random.seed(seed)

    nA = env.action_space.n
    nS = env.observation_space.n
    visit_counts_table = np.zeros(nS)

    # Q(s,a) -> each row is a different state and each columns represent a different action
    Q = np.zeros((nS, nA))

    rewards = []
    values = []
    q_tables = []
    policies = []
    times = [0]
    visit_counts = []
    converged = False
    ep = 0

    while ep <= num_episodes and not converged:
        delta = 0
        state = env.reset()

        start = clock()
        done = False
        tot_rew = 0
        if eps > 0.01:
            eps -= eps_decay

        while not done:
            action = eps_greedy(Q, state, eps)
            next_state, rew, done, _ = env.step(action)  # Take one step in the environment
            visit_counts_table[next_state] = 1

            delta = np.abs(lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action]))
            # get the max Q value for the next state
            Q[state][action] = Q[state][action] + lr * (rew + gamma * np.max(Q[next_state]) - Q[state][action])  # (4.6)

            # check if it's converged
            if ep > threshold_it and delta < threshold_val:
                converged = True
                break

            state = next_state
            tot_rew += rew

        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)

        ep += 1

        # test policy
        if ep == 0 or (ep % 500) == 0:
            # reward_test = get_reward(env, Q, num_games)
            rewards.append([ep, times[-1], tot_rew, delta])

            q_tables.append([ep, Q])

            visit_counts_table_str = str(visit_counts_table).replace('\n', '')
            visit_counts.append([ep, visit_counts_table_str])

            value = convert_q_2_value(Q)
            value_str = str(value).replace('\n', '')
            values.append([ep, value_str])

            policy = convert_q_2_policy(Q)
            policy_str = str(policy).replace('\n', '')
            policies.append([ep, policy_str])

            print('Iter:', ep, ' Counts:', visit_counts_table_str)

            # print('Iter:', ep, ' delta:', delta)

    # test policy
    if (ep % 500) != 0:
        # reward_test = get_reward(env, Q, num_games)
        rewards.append([ep, times[-1], tot_rew, delta])

        q_tables.append([ep, Q])

        value = convert_q_2_value(Q)
        value_str = str(value).replace('\n', '')
        values.append([ep, value_str])

        policy = convert_q_2_policy(Q)
        policy_str = str(policy).replace('\n', '')
        policies.append([ep, policy_str])

        print('Iter:', ep, ' Counts:', visit_counts_table_str)
        # print('Iter:', ep, ' delta:', delta)

    return rewards, values, policies, q_tables, visit_counts
