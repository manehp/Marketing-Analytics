from loguru import logger
from algorithms import EpsilonGreedy, ThompsonSampling, Visualization
import pandas as pd
import seaborn as sns


def comparison():
    """
    Run both algorithms, merge rewards into CSV, and plot results.
    """

    rewards = [1,2,3,4]
    trials = 20000

    # Run Epsilon-Greedy
    eps = EpsilonGreedy(rewards, n_trials=trials)
    eps.experiment()
    rewards_eps, regrets_eps, df_eps = eps.report()

    # Run Thompson Sampling
    th = ThompsonSampling(rewards, n_trials=trials)
    th.experiment()
    rewards_th, regrets_th, df_th = th.report()

    # Combine into one CSV
    df_combined = pd.concat([df_eps, df_th])
    df_combined.to_csv("all_rewards.csv", index=False)

    # Visualization
    vis = Visualization()
    vis.plot_learning(df_eps, "Epsilon-Greedy")
    vis.plot_learning(df_th, "Thompson Sampling")
    vis.plot1(rewards_eps, regrets_eps, rewards_th, regrets_th)

if __name__ == "__main__":
    logger.info("Starting Bandit Experiment")
    comparison()
    logger.success("Experiment Completed Successfully!")

