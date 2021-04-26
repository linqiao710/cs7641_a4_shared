
import time
from time import clock
import numpy as np


def convert_v_2_policy(env, nA, V, gamma):
    policy = np.zeros(len(V))
    for state in range(len(V)):
        policy[state] = np.argmax([eval_state_action(env, V, state, a, gamma) for a in range(nA)])
    return policy


def eval_state_action(env, V, s, a, gamma):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def run_episodes(env, nA, V, num_games, gamma):
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False

        while not done:
            # choose the best action using the value function
            action = np.argmax([eval_state_action(env, V, state, a, gamma) for a in range(nA)]) #(11)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    print('Won %i of %i games!'%(tot_rew, num_games))


def get_reward(env, nA, V, num_games, gamma, max_steps=800):
    state = env.reset()
    tot_rew = 0
    for i in range(num_games):
        done = False
        step = 0
        while not done and step <= max_steps:
            # choose the best action using the value function
            action = np.argmax([eval_state_action(env, V, state, a, gamma) for a in range(nA)])  # (11)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            tot_rew += reward
            step += 1
            if done:
                state = env.reset()

    return tot_rew


def value_iteration(env, gamma=0.99, eps=0.0001, num_games=100):
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n
    V = np.zeros(nS)

    rewards = []
    values = []
    policies = []
    times = [0]
    it = 0

    while True:
        delta = 0
        start = clock()

        # update the value for each state
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(env, V, s, a, gamma) for a in range(nA)])  # equation 3.10
            delta = max(delta, np.abs(old_v - V[s]))

        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)

        it += 1

        reward_test = get_reward(env, nA, V, num_games, gamma)
        rewards.append([it, times[-1], reward_test, delta])

        value_str = str(V).replace('\n', '')
        values.append([it, value_str])

        policy = convert_v_2_policy(env, nA, V, gamma)
        policy_str = str(policy).replace('\n', '')
        policies.append([it, policy_str])

        # if stable, break the cycle
        if delta < eps:
            break
        else:
            print('Iter:', it, ' delta:', np.round(delta, 5))

    return rewards, values, policies

