import gym
import numpy as np
from collections import deque

max_episodes = 100000

win_reward = 195.0

gamma = 1.0
alpha = 0.1
alpha_decay = 0.01
epsilon = 1.0
epsilon_decay = 0.0000001
episolon_min = 0.02


class Q_Learn:
    def __init__(self, gamma, alpha, alpha_decay, epsilon, episolon_decay, epsilon_min):
        # Variables
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q table
        self.q_table = dict()

    # Choose action either randomly or based on the Q table,
    # based on the factor epsilon
    def choose_action(self, env, state):
        if state in self.q_table:
            return env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax([self.q_table[state]])
        else:
            return env.action_space.sample()

    # Get the value of Q, returns zero if there is no value for the combination
    # of state and action, else return the recorded Q value
    def get_q(self, state, action):
        if not (state in self.q_table) or not (action in self.q_table[state]):
            return 0.0
        else:
            return self.q_table[state][action]

    # Updates the Q table, either adding new recordings of states and actions,
    # and the reward for these, or updating the value for recorded combinations
    def update_q(self, state, action, new_state, reward, env):
        if not (state in self.q_table):  # add new state
            self.q_table[state] = {}
        if not (action in self.q_table[state]):  # add new action to state
            self.q_table[state][action] = reward
        else:  # update Q value
            diff = self.alpha * \
                (reward+self.gamma
                 * max([self.get_q(new_state, a) for a in env.action_space]))
            new_q_value = (1-self.alpha)*self.q_table[state][action]+diff
            self.q_table[state][action] = new_q_value

        if self.epsilon > self.epsilon_min:  # reduce epsilon to rely more on Q
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':
    # setup agent and environment
    env = gym.make('CartPole-v1')
    agent = Q_Learn(gamma, alpha, alpha_decay, epsilon,
                    epsilon_decay, episolon_min)

    scores = deque(maxlen=100)  # continually records the latest 100 scores
    win = False

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        i = 0

        while not done:
            action = agent.choose_action(env, tuple(state))
            next_state, reward, done, _ = env.step(action)
            agent.update_q(tuple(state), action,
                           tuple(next_state), reward, env)
            state = next_state
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)
        if episode >= 100 and mean_score >= win_reward:
            print('Ran {} episodes. Solved after {} trials ✔'.format(
                episode+1, episode - 99))
            win = True
            break

    if not win:
        print('Did not solve after {} episodes'.format(episode+1))
