import copy
import random
import time
from time import clock
import numpy as np


def eval_state_action(env, V, s, a, gamma):
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def policy_evaluation(env, nS, V, policy, eps, gamma):
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = eval_state_action(env, V, s, policy[s], gamma)
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < eps:
            break
    return delta


def policy_improvement(env, nA, nS, V, policy, gamma):
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        policy[s] = np.argmax([eval_state_action(env, V, s, a, gamma) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False
    return policy_stable


def run_episodes(env, policy, num_games):
    tot_rew = 0
    state = env.reset()
    for _ in range(num_games):
        done = False
        while not done:
            next_state, reward, done, _ = env.step(policy[state])
            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()
    print('Won %i of %i games!' % (tot_rew, num_games))


def get_reward(env, policy, num_games, max_steps=800):
    state = env.reset()
    tot_rew = 0
    for i in range(num_games):
        done = False
        step = 0
        while not done and step <= max_steps:
            next_state, reward, done, _ = env.step(policy[state])
            # env.render()
            state = next_state
            tot_rew += reward
            step += 1
            if done:
                state = env.reset()

    return tot_rew


def policy_iteration(env, gamma=0.99, eps=0.0001, num_games=100):
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n
    V = np.zeros(nS)
    policy = np.zeros(nS)
    policy_stable = False

    rewards = []
    values = []
    policies = []
    times = [0]
    it = 0
    while not policy_stable:
        start = clock()
        delta = policy_evaluation(env, nS, V, policy, eps, gamma)
        policy_stable = policy_improvement(env, nA, nS, V, policy, gamma)

        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)

        it += 1

        reward_test = get_reward(env, policy, num_games)
        rewards.append([it, times[-1], reward_test, delta])

        value_str = str(V).replace('\n', '')
        values.append([it, value_str])

        policy_str = str(policy).replace('\n', '')
        policies.append([it, policy_str])

    print('Converged after %i policy iterations' % (it))

    return rewards, values, policies
