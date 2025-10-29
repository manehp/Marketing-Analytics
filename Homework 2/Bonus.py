'''
Better Implementation Plan (Bonus)

The improved implementation refactors the bandit experiment into a modular and scalable design.
A base class, MultiArmedBandit, handles common functionality such as logging rewards, tracking cumulative rewards,
and computing regrets. Child classes (EpsilonGreedy and ThompsonSampling) only implement their selection and update
strategies, reducing code duplication.
'''

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

sns.set(style="whitegrid")

# Bandit Base Class
class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass

# MultiArmedBandit Base Class
class MultiArmedBandit(Bandit):
    """Base class for multi-armed bandit algorithms"""
    def __init__(self, rewards, n_trials=20000):
        self.rewards = np.array(rewards)
        self.n_trials = n_trials
        self.n_bandits = len(rewards)
        self.total_reward = 0.0
        self.reward_log = []
        self.cumulative_rewards = np.zeros(n_trials)
        self.regrets = np.zeros(n_trials)
        self.optimal = max(rewards)

    def pull_strategy(self, t):
        raise NotImplementedError

    def update_strategy(self, bandit, reward):
        raise NotImplementedError

    def experiment(self):
        for t in range(self.n_trials):
            bandit = self.pull_strategy(t)
            reward = np.random.normal(self.rewards[bandit], 1)
            self.update_strategy(bandit, reward)
            self.total_reward += reward
            self.reward_log.append((bandit, reward))
            self.cumulative_rewards[t] = self.total_reward
            self.regrets[t] = self.optimal - self.rewards[bandit]

    def report(self, algorithm_name):
        df = pd.DataFrame(self.reward_log, columns=['Bandit', 'Reward'])
        df['Algorithm'] = algorithm_name
        df.to_csv(f"{algorithm_name.lower().replace(' ', '_')}_rewards.csv", index=False)
        print(f"{algorithm_name} Cumulative Reward: {self.total_reward:.3f}")
        print(f"{algorithm_name} Cumulative Regret: {self.regrets.sum():.3f}")
        return self.cumulative_rewards, np.cumsum(self.regrets), df

# Epsilon-Greedy Class
class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, rewards, n_trials=20000, initial_epsilon=1.0):
        super().__init__(rewards, n_trials)
        self.epsilon = initial_epsilon
        self.counts = np.zeros(self.n_bandits)
        self.values = np.zeros(self.n_bandits)

    def __repr__(self):
        return f"<EpsilonGreedy Bandits={self.n_bandits}, Trials={self.n_trials}>"

    def pull_strategy(self, t):
        self.epsilon = 1 / (t + 1)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_bandits - 1)
        return np.argmax(self.values)

    def update_strategy(self, bandit, reward):
        self.counts[bandit] += 1
        n = self.counts[bandit]
        self.values[bandit] += (1/n) * (reward - self.values[bandit])

# Thompson Sampling Class
class ThompsonSampling(MultiArmedBandit):
    def __init__(self, rewards, n_trials=20000, tau=1.0):
        super().__init__(rewards, n_trials)
        self.tau = tau
        self.means = np.zeros(self.n_bandits)
        self.lambdas = np.ones(self.n_bandits) * tau

    def __repr__(self):
        return f"<ThompsonSampling Bandits={self.n_bandits}, Trials={self.n_trials}>"

    def pull_strategy(self, t):
        samples = [np.random.normal(self.means[i], 1/np.sqrt(self.lambdas[i])) for i in range(self.n_bandits)]
        return np.argmax(samples)

    def update_strategy(self, bandit, reward):
        self.lambdas[bandit] += self.tau
        self.means[bandit] = (self.means[bandit] * (self.lambdas[bandit] - self.tau) + reward * self.tau) / self.lambdas[bandit]

# Visualization Class
class Visualization:
    def plot_comparison(self, rewards_eps, regrets_eps, rewards_th, regrets_th):
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(rewards_eps, label='Epsilon-Greedy')
        plt.plot(rewards_th, label='Thompson Sampling')
        plt.xlabel('Trials'); plt.ylabel('Cumulative Reward')
        plt.legend(); plt.title('Cumulative Rewards')

        plt.subplot(1,2,2)
        plt.plot(regrets_eps, label='Epsilon-Greedy')
        plt.plot(regrets_th, label='Thompson Sampling')
        plt.xlabel('Trials'); plt.ylabel('Cumulative Regret')
        plt.legend(); plt.title('Cumulative Regrets')

        plt.tight_layout(); plt.show()

    def plot_learning(self, df, algorithm_name):
        plt.figure(figsize=(10,5))
        for bandit in df['Bandit'].unique():
            rewards = df[df['Bandit']==bandit]['Reward'].expanding().mean()
            plt.plot(rewards, label=f'Bandit {bandit}')
        plt.title(f'Learning Process: {algorithm_name}')
        plt.xlabel('Trial'); plt.ylabel('Average Reward'); plt.legend(); plt.show()

# Main Experiment
def run_comparison():
    rewards = [1,2,3,4]
    n_trials = 20000

    # Epsilon-Greedy
    eps = EpsilonGreedy(rewards, n_trials)
    eps.experiment()
    rewards_eps, regrets_eps, df_eps = eps.report("Epsilon-Greedy")

    # Thompson Sampling
    th = ThompsonSampling(rewards, n_trials)
    th.experiment()
    rewards_th, regrets_th, df_th = th.report("Thompson Sampling")

    # Combined CSV
    pd.concat([df_eps, df_th]).to_csv("all_rewards_bonus.csv", index=False)

    # Visualization
    vis = Visualization()
    vis.plot_learning(df_eps, "Epsilon-Greedy")
    vis.plot_learning(df_th, "Thompson Sampling")
    vis.plot_comparison(rewards_eps, regrets_eps, rewards_th, regrets_th)

if __name__ == "__main__":
    run_comparison()
