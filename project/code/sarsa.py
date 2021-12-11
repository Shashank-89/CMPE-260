import collections

import gym
import random
import numpy as np
import matplotlib
import logging

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Since the range of values are continuous from -inf to +inf
# we need to discretize the state,
# State description:
# s[0] is the horizontal coordinate
# s[1] is the vertical coordinate
# s[2] is the horizontal speed
# s[3] is the vertical speed
# s[4] is the angle
# s[5] is the angular speed
# s[6] 1 if first leg has contact, else 0
# s[7] 1 if second leg has contact, else 0
# Refs :


def env_info_test(env_name='LunarLander-v2'):
    env = gym.make(env_name)
    env.reset()
    # printing details about the environment
    print(f'action space :{env.action_space}')
    print(f'action space eg sample : {env.action_space.sample()}')
    print(f'obs space : {env.observation_space}')
    print(f'obs space eg sample : {env.observation_space.sample}')
    print(f'obs space high : {env.observation_space.high}')
    print(f'obs space eg sample : {env.observation_space.low}')
    print(env.reward_range)

    # for test run to render the env
    # Since each value of velocity and coordinates is in range -inf to +inf, we need to
    # find a practical possible ranges for each of the states. Trying out each action and recording max value reached
    # before the end of episode to guage the practical range for descritization of space

    for action in range(env.action_space.n):
        n = 0
        for steps in range(100):
            env.render()
            n = n + 1
            state, reward, done, info = env.step(action)
            if steps % 25 == 0:
                print(f'state : \n{state}')
                print(f'action : {action}')
                print(f'reward : {reward}')
            if done:
                print(n)
                break
        env.reset()
    env.close()


def discretize_state(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))),
                      min(2, max(-2, int((state[1]) / 0.1))),
                      min(2, max(-2, int((state[2]) / 0.1))),
                      min(2, max(-2, int((state[3]) / 0.1))),
                      min(2, max(-2, int((state[4]) / 0.1))),
                      min(2, max(-2, int((state[5]) / 0.1))),
                      int(state[6]),
                      int(state[7]))

    return discrete_state


# assuming discrete actions labeled from 0 to n
def choose_action(q_func, epsilon, curr_state, env):
    num_actions = env.action_space.n
    if np.random.random() < epsilon:
        return random.choice(range(num_actions))
    else:
        qvals = [q_func[curr_state + (action,)] for action in range(num_actions)]
        return np.argmax(qvals)


# Exploration vs Exploitation parameter
def decay_epsilon(curr_eps, exploration_final_eps):
    if curr_eps < exploration_final_eps:
        return curr_eps
    return curr_eps * 0.99


def plot_rewards(reward_per_ep, num_episodes, plot_per_n_ep=100):
    intervals = int(num_episodes / plot_per_n_ep)
    l = []
    for i in range(intervals):
        l.append(round(np.mean(reward_per_ep[i * 100: (i + 1) * 100]), 1))
    plt.plot(range(0, num_episodes, plot_per_n_ep), l)
    plt.xlabel("Episodes")
    plt.ylabel("Reward per {} episodes".format(plot_per_n_ep))
    plt.title("RL Lander(s)")
    plt.legend('SARSA', loc="lower right")
    plt.show()


def sarsa_lunar_lander(num_episodes, gamma, lr, min_eps):
    q_function = collections.defaultdict(float)
    log_freq = 500
    epsilon = 1.0  # Exploration vs Exploitation parameter
    reward_per_ep = [0.0]
    env = gym.make('LunarLander-v2')
    for i in range(num_episodes):
        t = 0  # timesteps
        curr_state = discretize_state(env.reset())
        action = choose_action(q_function, epsilon, curr_state, env)

        # loop until done for this episode
        while True:
            # S-->A-->R-->S'
            observation, reward, done, _ = env.step(action)
            next_state = discretize_state(observation)
            # choose next action to take from q_function
            next_action = choose_action(q_function, epsilon, next_state, env)

            # update policy (q_function)
            if not done:
                q_function[curr_state + (action,)] += lr * (
                        reward + gamma * q_function[next_state + (next_action,)] - q_function[
                    curr_state + (action,)])
            else:
                # end of episode so no next action or next state
                q_function[curr_state + (action,)] += lr * (reward - q_function[curr_state + (action,)])

            reward_per_ep[-1] += reward

            if done:
                if (i + 1) % log_freq == 0:
                    print("\nEpisode finished after {} timesteps".format(t + 1))
                    print("Episode {}: Total Return = {}".format(i + 1, reward_per_ep[-1]))
                    print("Total keys in q_states dictionary = {}".format(len(q_function)))

                if (i + 1) % 100 == 0:
                    mean_reward_for_100eps = round(np.mean(reward_per_ep[-101:-1]), 1)
                    print("mean reward for last 100 episodes: {}".format(mean_reward_for_100eps))

                epsilon = decay_epsilon(epsilon, min_eps)
                reward_per_ep.append(0.0)  # appending reward for next episode
                break

            curr_state = next_state
            action = next_action
            t += 1
    env.close()
    return q_function, reward_per_ep


def record(q_function):
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, "recording", force=True)

    curr_state = discretize_state(env.reset())
    num_actions = env.action_space.n
    while True:
        env.render()
        qvals = [q_function[curr_state + (action,)] for action in range(num_actions)]
        action = np.argmax(qvals)
        obs, rew, done, info = env.step(action)
        curr_state = discretize_state(obs)
        if done:
            break
    env.close()


if __name__ == "__main__":
    q_function, reward_per_ep = sarsa_lunar_lander(num_episodes=10000, gamma=0.99, lr=0.1, min_eps=0.01)
    plot_rewards(reward_per_ep, 10000, 100)
    record(q_function)

