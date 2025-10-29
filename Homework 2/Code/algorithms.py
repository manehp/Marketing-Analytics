import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from loguru import logger
from Bandit import Bandit

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy algorithm with 1/t decay.

        Attributes:
        rewards (list): True mean rewards of each bandit.
        n_bandits (int): Number of bandits.
        n_trials (int): Number of experiments to run.
        epsilon (float): Exploration probability, decays as 1/t.
        counts (np.ndarray): Number of times each bandit is pulled.
        values (np.ndarray): Estimated mean reward for each bandit.
        total_reward (float): Total accumulated reward.
        cumulative_rewards (list): Cumulative reward at each trial.
        regrets (list): Regret at each trial.
        reward_log (list): Stores all (bandit, reward) tuples for CSV.

    """


    def __init__(self, rewards, n_trials=20000, epsilon=1.0):

        """
        Initialize the epsilon-greedy algorithm with parameters.

        Args:
            rewards (list): True reward values for each bandit.
            n_trials (int): Number of trials to simulate.
            epsilon (float): Initial exploration probability.
        """
        self.rewards = rewards
        self.n_bandits = len(rewards)
        self.n_trials = n_trials
        self.epsilon = epsilon
        self.counts = np.zeros(self.n_bandits)
        self.values = np.zeros(self.n_bandits)
        self.total_reward = 0
        self.cumulative_rewards = []
        self.regrets = []
        self.optimal = max(rewards)
        self.reward_log = []

    def __repr__(self):
        return f"<EpsilonGreedy Bandits={self.n_bandits}, Trials={self.n_trials}>"

    def pull(self):
        """
        Select a bandit using epsilon-greedy strategy.
        Exploit the best bandit with probability 1-epsilon, otherwise explore randomly.

        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_bandits - 1)
        return np.argmax(self.values)

    def update(self, chosen_bandit, reward):
        """
        Update estimated value for the chosen bandit.

        Args:
            chosen_bandit (int): Index of selected bandit.
            reward (float): Observed reward.
        """

        self.counts[chosen_bandit] += 1
        n = self.counts[chosen_bandit]
        self.values[chosen_bandit] += (1 / n) * (reward - self.values[chosen_bandit])

    def experiment(self):
        """

        Run the epsilon-greedy experiment over n_trials.
        Updates cumulative rewards, regrets, and logs individual rewards.

        """
        logger.info("Running Epsilon-Greedy experiment...")
        for t in range(1, self.n_trials + 1):
            self.epsilon = 1 / t
            bandit = self.pull()
            reward = np.random.normal(self.rewards[bandit], 1)
            self.update(bandit, reward)
            self.total_reward += reward
            self.cumulative_rewards.append(self.total_reward)
            self.regrets.append(self.optimal - self.rewards[bandit])
            self.reward_log.append((bandit, reward))

    def report(self):
        """
        Generate a report, save rewards CSV, print cumulative metrics.

        Returns:
            tuple: cumulative_rewards, cumulative_regrets, reward_log_df
        """
        df = pd.DataFrame(self.reward_log, columns=['Bandit','Reward'])
        df['Algorithm'] = "Epsilon-Greedy"
        df.to_csv("epsilon_greedy_rewards.csv", index=False)

        cum_reward = self.total_reward
        cum_regret = sum(self.regrets)
        print(f"Epsilon-Greedy Cumulative Reward: {cum_reward:.3f}")
        print(f"Epsilon-Greedy Cumulative Regret: {cum_regret:.3f}")

        return self.cumulative_rewards, np.cumsum(self.regrets), df


class ThompsonSampling(Bandit):
    """
    Thompson Sampling multi-armed bandit algorithm.

    Attributes:
        rewards (list): True mean rewards of each bandit.
        n_bandits (int): Number of bandits.
        n_trials (int): Number of experiments to run.
        tau (float): Precision of reward distribution.
        means (np.ndarray): Posterior mean for each bandit.
        lambdas (np.ndarray): Posterior precision for each bandit.
        total_reward (float): Total accumulated reward.
        cumulative_rewards (list): Cumulative reward at each trial.
        regrets (list): Regret at each trial.
        reward_log (list): Stores all (bandit, reward) tuples for CSV.
    """
    def __init__(self, rewards, n_trials=20000, tau=1.0):
        self.rewards = rewards
        self.n_bandits = len(rewards)
        self.n_trials = n_trials
        self.tau = tau
        self.total_reward = 0
        self.means = np.zeros(self.n_bandits)
        self.lambdas = np.ones(self.n_bandits) * self.tau
        self.cumulative_rewards = []
        self.regrets = []
        self.optimal = max(rewards)
        self.reward_log = []

    def __repr__(self):
        return f"<ThompsonSampling Bandits={self.n_bandits}, Trials={self.n_trials}>"

    def pull(self):
        """
        Sample from posterior for each bandit and select the max.
        """
        samples = [np.random.normal(self.means[i], 1 / np.sqrt(self.lambdas[i])) for i in range(self.n_bandits)]
        return np.argmax(samples)

    def update(self, bandit, reward):
        """
        Update posterior mean and precision for the chosen bandit.

        Args:
            bandit (int): Index of chosen bandit.
            reward (float): Observed reward.
        """
        self.lambdas[bandit] += self.tau
        self.means[bandit] = (self.means[bandit] * (self.lambdas[bandit] - self.tau) + reward * self.tau) / self.lambdas[bandit]

    def experiment(self):
        """
        Run Thompson Sampling experiment over n_trials.
        """
        logger.info("Running Thompson Sampling experiment...")
        for _ in range(self.n_trials):
            bandit = self.pull()
            reward = np.random.normal(self.rewards[bandit], 1)
            self.update(bandit, reward)
            self.total_reward += reward
            self.cumulative_rewards.append(self.total_reward)
            self.regrets.append(self.optimal - self.rewards[bandit])
            self.reward_log.append((bandit, reward))

    def report(self):
        """
        Generate report, save CSV, print cumulative metrics.

        Returns:
            tuple: cumulative_rewards, cumulative_regrets, reward_log_df
        """
        df = pd.DataFrame(self.reward_log, columns=['Bandit','Reward'])
        df['Algorithm'] = "Thompson Sampling"
        df.to_csv("thompson_sampling_rewards.csv", index=False)

        cum_reward = self.total_reward
        cum_regret = sum(self.regrets)
        print(f"Thompson Sampling Cumulative Reward: {cum_reward:.3f}")
        print(f"Thompson Sampling Cumulative Regret: {cum_regret:.3f}")

        return self.cumulative_rewards, np.cumsum(self.regrets), df


class Visualization:
    """
    Visualization helper class for plotting bandit performance.
    """
    def plot1(self, rewards_eps, regrets_eps, rewards_th, regrets_th):
        """
        Args:
            rewards_eps (list or np.ndarray): Cumulative rewards from Epsilon-Greedy.
            regrets_eps (list or np.ndarray): Cumulative regrets from Epsilon-Greedy.
            rewards_th (list or np.ndarray): Cumulative rewards from Thompson Sampling.
            regrets_th (list or np.ndarray): Cumulative regrets from Thompson Sampling.

        Returns:
            None. Displays the plot using matplotlib.

        """
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_eps, label='Epsilon-Greedy')
        plt.plot(rewards_th, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.title('Cumulative Rewards')

        plt.subplot(1, 2, 2)
        plt.plot(regrets_eps, label='Epsilon-Greedy')
        plt.plot(regrets_th, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.title('Cumulative Regrets')

        plt.tight_layout()
        plt.show()

    def plot_learning(self, reward_df, algorithm_name):
        """
        Plot learning process (average reward per bandit over time).

        Args:
            reward_df (pd.DataFrame): DataFrame with Bandit, Reward, Algorithm columns.
            algorithm_name (str): Algorithm name for the title.
        """
        import seaborn as sns
        sns.set(style="whitegrid")
        plt.figure(figsize=(10,5))
        for bandit in reward_df['Bandit'].unique():
            rewards = reward_df[reward_df['Bandit']==bandit]['Reward'].expanding().mean()
            plt.plot(rewards, label=f'Bandit {bandit}')
        plt.title(f'Learning Process: {algorithm_name}')
        plt.xlabel('Trial')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.show()
