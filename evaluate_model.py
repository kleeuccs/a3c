"""
Generated code with interative prompts using
ChatGPT 40

Render Mode: The warning you received suggests specifying a render mode. In this example, I've set render_mode="human" when creating the environment. This will display the environment's GUI window.

Plotting with Matplotlib:

Import Matplotlib: Added import matplotlib.pyplot as plt to handle plotting.
Plot Total Rewards: After evaluating all episodes, the script plots the total rewards for each episode using plt.plot(total_rewards).
Labels and Title: Added labels and a title to the plot for clarity.
Visualization: The plot will show how the total rewards change over episodes, giving you a visual representation of the learning progress.
Interpreting the results of training and evaluation in an A3C (Asynchronous Advantage Actor-Critic) setup involves analyzing the rewards obtained during training and evaluation phases. Here's how you can interpret these results and identify learning trends:

Training Phase
Total Reward per Episode:

Definition: The total reward accumulated by the agent during a single episode.
Interpretation: Higher total rewards generally indicate better performance. As training progresses, you should see an upward trend in the total rewards if the agent is learning effectively.
Average Reward:

Definition: The average of total rewards over a set number of episodes.
Interpretation: A rising average reward over time suggests that the agent is improving its policy and learning to maximize rewards.
Reward Variability:

Definition: The fluctuation in rewards between episodes.
Interpretation: High variability might indicate that the agent is still exploring different strategies. As learning stabilizes, variability should decrease, showing more consistent performance.
Evaluation Phase
Consistent Performance:

Definition: The agent consistently achieves high rewards across multiple evaluation episodes.
Interpretation: Consistent high rewards indicate that the agent has learned a robust policy that generalizes well to the environment.
Comparison to Baseline:

Definition: Comparing the agent's performance to a baseline or random policy.
Interpretation: If the agent significantly outperforms a baseline, it suggests effective learning. The baseline could be a random policy or a simple heuristic.
Plateauing of Rewards:

Definition: The rewards reach a stable level and do not increase further.
Interpretation: This could indicate that the agent has reached its optimal policy given the current setup, or it might be stuck in a local optimum.
Identifying Learning Trends
Upward Trend in Rewards: Indicates effective learning and improvement in the agent's policy.
Decreasing Variability: Suggests that the agent is stabilizing its learning and becoming more consistent.
Plateauing: May require adjustments in hyperparameters, exploration strategies, or model architecture to push past local optima.
Visualization
Plotting Rewards: Visualizing the total and average rewards over episodes can help identify trends and assess the learning progress.
Moving Average: Applying a moving average to the reward plot can smooth out short-term fluctuations and highlight the overall trend.
By analyzing these metrics and trends, you can assess the effectiveness of your A3C implementation and make informed decisions about potential improvements or adjustments needed in your training setup.
"""

import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from actor_critic_model import ActorCritic

def evaluate_trained_model(model_path, env_name='CartPole-v1', num_episodes=100):
    # Load the trained model
    model = ActorCritic(input_dim=4, action_space=2)  # Adjust input_dim and action_space as needed
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize the environment with render mode
    env = gym.make(env_name, render_mode="human")

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()  # Render the environment for visualization
            state_tensor = torch.from_numpy(state).float()
            policy_dist, _ = model(state_tensor)
            action = torch.argmax(policy_dist).item()

            state, reward, done, _, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")

    # Plotting the rewards
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards over Episodes')
    plt.show()

if __name__ == "__main__":
    evaluate_trained_model('trained_model.pth')

