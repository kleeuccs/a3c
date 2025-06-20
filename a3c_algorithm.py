"""
Generated code with interative prompts using
ChatGPT 40

Instructions
Training:

Run a3c_algorithm.py to train the model. This will save the model's state dictionary to trained_model.pth.
Evaluation:

Run evaluate_model.py to evaluate the trained model. Ensure the path to the saved model is correct.

Dependencies: pip install gymnasium torch
"""
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from actor_critic_model import ActorCritic

# Initialize the model and optimizer
env = gym.make('CartPole-v1')
model = ActorCritic(input_dim=env.observation_space.shape[0], action_space=env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track rewards for plotting
total_rewards = []

def train_a3c():
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.from_numpy(state).float()
        policy_dist, value = model(state_tensor)
        action = torch.multinomial(policy_dist, 1).item()
        next_state, reward, done, _, _ = env.step(action)

        # Compute advantage and update model
        _, next_value = model(torch.from_numpy(next_state).float())
        advantage = reward + (1 - done) * next_value.item() - value.item()

        # Calculate loss
        policy_loss = -torch.log(policy_dist[action]) * advantage
        value_loss = F.mse_loss(value, torch.tensor([reward + (1 - done) * next_value.item()]))
        loss = policy_loss + value_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)

if __name__ == "__main__":
    start_time = time.time()
    total_episodes = 1000
    print("Total Episodes to train:", total_episodes)

    for i in range(total_episodes):
        if i % 100 == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"{i} episodes completed... {duration:.2f} seconds elapsed.")

        train_a3c()

    end_time = time.time()
    duration = end_time - start_time
    print("Total time: ", duration, "seconds.")

    # Save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')

    # Plotting the training rewards
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()