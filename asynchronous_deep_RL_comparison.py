import gymnasium as gym
import numpy as np
import tensorflow as tf
import threading
import multiprocessing
import matplotlib.pyplot as plt
import time
import pandas as pd
from collections import deque
import os

# Ensure TensorFlow operations are deterministic for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global parameters
ENV_NAME = "CartPole-v1"
NUM_WORKERS = multiprocessing.cpu_count() - 1  # Leave one CPU free
MAX_GLOBAL_EPISODES = 500
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic
N_STEP_RETURN = 5  # for n-step algorithms
EVAL_INTERVAL = 25  # Evaluate every N episodes
EVAL_EPISODES = 10  # Number of episodes for evaluation

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Sample environment to get state and action dimensions
env = gym.make(ENV_NAME)
N_S = env.observation_space.shape[0]
if isinstance(env.action_space, gym.spaces.Discrete):
    N_A = env.action_space.n
else:
    N_A = env.action_space.shape[0]
env.close()

# Build neural network
def build_network(scope, trainable=True):
    with tf.variable_scope(scope):
        # Shared network layers
        kernel_init = tf.random_normal_initializer(0., .1)
        
        inputs = tf.keras.layers.Input(shape=(N_S,), name='state_input')
        dense1 = tf.keras.layers.Dense(128, activation='relu', 
                                      kernel_initializer=kernel_init)(inputs)
        dense2 = tf.keras.layers.Dense(64, activation='relu', 
                                      kernel_initializer=kernel_init)(dense1)
        
        # Actor network (policy)
        actor_output = tf.keras.layers.Dense(N_A, activation='softmax', 
                                           kernel_initializer=kernel_init,
                                           name='actor_output')(dense2)
        
        # Critic network (value)
        critic_output = tf.keras.layers.Dense(1, kernel_initializer=kernel_init,
                                            name='critic_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, 
                             outputs=[actor_output, critic_output])
        return model

# Evaluation function to test model performance
def evaluate_model(model, num_episodes=10):
    eval_env = gym.make(ENV_NAME, render_mode=None)
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use the model to select the best action
            action_probs, _ = model(np.array([state]))
            action = np.argmax(action_probs.numpy()[0])  # Greedy policy for evaluation
            
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            
        rewards.append(episode_reward)
    
    eval_env.close()
    return np.mean(rewards)

# Base Worker class
class Worker(threading.Thread):
    def __init__(self, name, global_model, global_actor_optimizer, global_critic_optimizer, 
                 global_episodes, result_queue, lock):
        super(Worker, self).__init__()
        self.name = name
        self.global_model = global_model
        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer
        self.global_episodes = global_episodes
        self.result_queue = result_queue
        self.lock = lock
        
        self.local_model = build_network(name)
        self.env = gym.make(ENV_NAME)
        
        # Episode reward history
        self.ep_reward = 0
        
    def run(self):
        total_step = 1
        while self.global_episodes.numpy() < MAX_GLOBAL_EPISODES:
            # Sync with global network
            self.local_model.set_weights(self.global_model.get_weights())
            
            # Collect experience based on algorithm type
            self.ep_reward = 0
            self._collect_experience_and_update()
            
            # Push episode reward to the queue
            self.result_queue.put(self.ep_reward)
            
            # Update global episode counter
            with self.lock:
                self.global_episodes.assign_add(1)
            
            if total_step % 100 == 0:
                print(f"Worker {self.name}, Episode: {self.global_episodes.numpy()}, Reward: {self.ep_reward}")
            
            total_step += 1
    
    def _collect_experience_and_update(self):
        # This method will be overridden by specific algorithm implementations
        pass

# Asynchronous One-step Sarsa Worker
class SarsaWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        
        # Choose initial action using epsilon-greedy
        action_probs, _ = self.local_model(np.array([state]))
        action = self._epsilon_greedy_action(action_probs.numpy()[0])
        
        while not done:
            # Take action and observe
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.ep_reward += reward
            
            # Choose next action using epsilon-greedy
            next_action_probs, _ = self.local_model(np.array([next_state]))
            next_action = self._epsilon_greedy_action(next_action_probs.numpy()[0])
            
            # Calculate TD target for Sarsa: r + γQ(s',a')
            with tf.GradientTape() as tape:
                action_probs, value = self.local_model(np.array([state]))
                next_action_probs, next_value = self.local_model(np.array([next_state]))
                
                # Sarsa target uses the actual next action
                target = reward + (0 if done else GAMMA * next_value)
                advantage = target - value
                
                # Actor loss (policy gradient with advantage)
                action_oh = tf.one_hot(action, N_A, dtype=tf.float32)
                log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh) + 1e-10)
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10))
                actor_loss = -log_prob * tf.stop_gradient(advantage) - ENTROPY_BETA * entropy
                
                # Critic loss (MSE)
                critic_loss = tf.square(advantage)
                
                # Total loss
                total_loss = actor_loss + critic_loss
            
            # Calculate gradients and apply to global network
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            
            # Apply gradients to global model
            self.global_actor_optimizer.apply_gradients(
                zip(grads[:len(self.global_model.get_layer('actor_output').trainable_weights)],
                    self.global_model.get_layer('actor_output').trainable_weights))
            
            self.global_critic_optimizer.apply_gradients(
                zip(grads[len(self.global_model.get_layer('actor_output').trainable_weights):],
                    self.global_model.get_layer('critic_output').trainable_weights))
            
            # Move to next state and action
            state = next_state
            action = next_action
    
    def _epsilon_greedy_action(self, action_probs, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(N_A)
        else:
            return np.random.choice(N_A, p=action_probs)

# Asynchronous One-step Q-Learning Worker
class QLearningWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        
        while not done:
            # Choose action using epsilon-greedy
            action_probs, _ = self.local_model(np.array([state]))
            action = self._epsilon_greedy_action(action_probs.numpy()[0])
            
            # Take action and observe
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.ep_reward += reward
            
            # Calculate TD target for Q-learning: r + γ max_a' Q(s',a')
            with tf.GradientTape() as tape:
                action_probs, value = self.local_model(np.array([state]))
                next_action_probs, next_value = self.local_model(np.array([next_state]))
                
                # Q-learning uses max Q-value for next state
                target = reward + (0 if done else GAMMA * next_value)
                advantage = target - value
                
                # Actor loss
                action_oh = tf.one_hot(action, N_A, dtype=tf.float32)
                log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh) + 1e-10)
                entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10))
                actor_loss = -log_prob * tf.stop_gradient(advantage) - ENTROPY_BETA * entropy
                
                # Critic loss
                critic_loss = tf.square(advantage)
                
                # Total loss
                total_loss = actor_loss + critic_loss
            
            # Calculate gradients and apply to global network
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            
            # Apply gradients to global model
            self.global_actor_optimizer.apply_gradients(
                zip(grads[:len(self.global_model.get_layer('actor_output').trainable_weights)],
                    self.global_model.get_layer('actor_output').trainable_weights))
            
            self.global_critic_optimizer.apply_gradients(
                zip(grads[len(self.global_model.get_layer('actor_output').trainable_weights):],
                    self.global_model.get_layer('critic_output').trainable_weights))
            
            # Move to next state
            state = next_state
    
    def _epsilon_greedy_action(self, action_probs, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(N_A)
        else:
            return np.random.choice(N_A, p=action_probs)

# Asynchronous n-step Q-Learning Worker
class NStepQLearningWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        
        while not done:
            buffer_s, buffer_a, buffer_r = [], [], []
            
            # Synchronize with global network
            self.local_model.set_weights(self.global_model.get_weights())
            
            # Collect n-step experience
            for _ in range(N_STEP_RETURN):
                if done:
                    break
                
                # Choose action using epsilon-greedy
                action_probs, _ = self.local_model(np.array([state]))
                action = self._epsilon_greedy_action(action_probs.numpy()[0])
                
                # Take action and observe
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.ep_reward += reward
                
                # Store transition
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                
                state = next_state
            
            # Calculate n-step return
            if done:
                v_s_ = 0
            else:
                _, v_s_ = self.local_model(np.array([state]))
                v_s_ = v_s_.numpy()[0, 0]
            
            # Compute n-step returns
            buffer_v_target = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                buffer_v_target.append(v_s_)
            buffer_v_target.reverse()
            
            # Update global network
            with tf.GradientTape() as tape:
                loss = 0
                for i in range(len(buffer_s)):
                    action_probs, value = self.local_model(np.array([buffer_s[i]]))
                    advantage = buffer_v_target[i] - value
                    
                    # Actor loss
                    action_oh = tf.one_hot(buffer_a[i], N_A, dtype=tf.float32)
                    log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh) + 1e-10)
                    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10))
                    actor_loss = -log_prob * tf.stop_gradient(advantage) - ENTROPY_BETA * entropy
                    
                    # Critic loss
                    critic_loss = tf.square(advantage)
                    
                    # Add to total loss
                    loss += actor_loss + critic_loss
            
            # Calculate gradients and apply to global network
            grads = tape.gradient(loss, self.local_model.trainable_weights)
            
            # Apply gradients to global model
            self.global_actor_optimizer.apply_gradients(
                zip(grads[:len(self.global_model.get_layer('actor_output').trainable_weights)],
                    self.global_model.get_layer('actor_output').trainable_weights))
            
            self.global_critic_optimizer.apply_gradients(
                zip(grads[len(self.global_model.get_layer('actor_output').trainable_weights):],
                    self.global_model.get_layer('critic_output').trainable_weights))
    
    def _epsilon_greedy_action(self, action_probs, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(N_A)
        else:
            return np.random.choice(N_A, p=action_probs)

# Asynchronous Advantage Actor-Critic (A3C) Worker
class A3CWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        
        while not done:
            buffer_s, buffer_a, buffer_r = [], [], []
            
            # Synchronize with global network
            self.local_model.set_weights(self.global_model.get_weights())
            
            # Collect n-step experience
            for _ in range(N_STEP_RETURN):
                if done:
                    break
                
                # Choose action using policy
                action_probs, _ = self.local_model(np.array([state]))
                action = np.random.choice(N_A, p=action_probs.numpy()[0])
                
                # Take action and observe
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.ep_reward += reward
                
                # Store transition
                buffer_s.append(state)
                buffer_a.append(action)
                buffer_r.append(reward)
                
                state = next_state
            
            # Calculate advantage and value targets
            if done:
                v_s_ = 0
            else:
                _, v_s_ = self.local_model(np.array([state]))
                v_s_ = v_s_.numpy()[0, 0]
            
            # Compute n-step returns and advantages
            buffer_v_target = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                buffer_v_target.append(v_s_)
            buffer_v_target.reverse()
            
            # Update global network
            with tf.GradientTape() as tape:
                actor_loss = 0
                critic_loss = 0
                
                for i in range(len(buffer_s)):
                    action_probs, value = self.local_model(np.array([buffer_s[i]]))
                    advantage = buffer_v_target[i] - value
                    
                    # Actor loss (policy gradient with advantage)
                    action_oh = tf.one_hot(buffer_a[i], N_A, dtype=tf.float32)
                    log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh) + 1e-10)
                    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10))
                    actor_loss += -log_prob * tf.stop_gradient(advantage) - ENTROPY_BETA * entropy
                    
                    # Critic loss (MSE)
                    critic_loss += tf.square(advantage)
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss
            
            # Calculate gradients and apply to global network
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)
            
            # Apply gradients to global model
            self.global_actor_optimizer.apply_gradients(
                zip(grads[:len(self.global_model.get_layer('actor_output').trainable_weights)],
                    self.global_model.get_layer('actor_output').trainable_weights))
            
            self.global_critic_optimizer.apply_gradients(
                zip(grads[len(self.global_model.get_layer('actor_output').trainable_weights):],
                    self.global_model.get_layer('critic_output').trainable_weights))

# Function to train a specific algorithm and return results
def train_algorithm(algorithm_name):
    print(f"Training {algorithm_name}...")
    
    # Reset TensorFlow graph
    tf.keras.backend.clear_session()
    
    # Create global network
    global_model = build_network(f"{algorithm_name}_Global")
    
    # Create optimizers
    global_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_A)
    global_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_C)
    
    # Create global episode counter
    global_episodes = tf.Variable(0, dtype=tf.int32, trainable=False)
    
    # Create result queue and lock
    result_queue = multiprocessing.Queue()
    lock = threading.Lock()
    
    # Create workers based on algorithm choice
    workers = []
    for i in range(NUM_WORKERS):
        worker_name = f'{algorithm_name}_Worker_{i}'
        
        if algorithm_name == 'Sarsa':
            worker = SarsaWorker(worker_name, global_model, 
                                global_actor_optimizer, global_critic_optimizer,
                                global_episodes, result_queue, lock)
        elif algorithm_name == 'QLearning':
            worker = QLearningWorker(worker_name, global_model, 
                                    global_actor_optimizer, global_critic_optimizer,
                                    global_episodes, result_queue, lock)
        elif algorithm_name == 'NStepQLearning':
            worker = NStepQLearningWorker(worker_name, global_model, 
                                        global_actor_optimizer, global_critic_optimizer,
                                        global_episodes, result_queue, lock)
        else:  # Default to A3C
            worker = A3CWorker(worker_name, global_model, 
                              global_actor_optimizer, global_critic_optimizer,
                              global_episodes, result_queue, lock)
        
        workers.append(worker)
    
    # Start workers
    for worker in workers:
        worker.daemon = True
        worker.start()
    
    # Lists to store results
    episode_rewards = []
    evaluation_scores = []
    evaluation_episodes = []
    
    # Collect results and evaluate periodically
    while global_episodes.numpy() < MAX_GLOBAL_EPISODES:
        episode_reward = result_queue.get()
        episode_rewards.append(episode_reward)
        
        # Evaluate model periodically
        if global_episodes.numpy() % EVAL_INTERVAL == 0:
            eval_score = evaluate_model(global_model, EVAL_EPISODES)
            evaluation_scores.append(eval_score)
            evaluation_episodes.append(global_episodes.numpy())
            print(f"{algorithm_name} - Episode {global_episodes.numpy()}: Evaluation score = {eval_score}")
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join()
    
    # Save model
    global_model.save_weights(f'results/{algorithm_name}_model.h5')
    
    # Return results
    return {
        'episode_rewards': episode_rewards,
        'evaluation_scores': evaluation_scores,
        'evaluation_episodes': evaluation_episodes,
        'model': global_model
    }

# Function to plot training and evaluation results
def plot_results(results):
    plt.figure(figsize=(20, 10))
    
    # Plot 1: Training rewards
    plt.subplot(1, 2, 1)
    for algo, data in results.items():
        # Smooth rewards with rolling average
        rewards = pd.Series(data['episode_rewards']).rolling(window=20).mean()
        plt.plot(rewards, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (20-episode rolling mean)')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Evaluation scores
    plt.subplot(1, 2, 2)
    for algo, data in results.items():
        plt.plot(data['evaluation_episodes'], data['evaluation_scores'], 'o-', label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Evaluation Score (avg over 10 episodes)')
    plt.title('Evaluation Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/algorithm_comparison.png')
    plt.show()

# Function to compare algorithms by time to reach a target score
def plot_learning_efficiency(results, target_score=195):
    plt.figure(figsize=(12, 6))
    
    # Find episodes needed to reach target score
    episodes_to_target = {}
    for algo, data in results.items():
        eval_scores = data['evaluation_scores']
        eval_episodes = data['evaluation_episodes']
        
        # Find first episode where evaluation score >= target
        for i, score in enumerate(eval_scores):
            if score >= target_score:
                episodes_to_target[algo] = eval_episodes[i]
                break
        
        # If target never reached, use max episodes
        if algo not in episodes_to_target:
            episodes_to_target[algo] = MAX_GLOBAL_EPISODES
    
    # Plot bar chart
    plt.bar(episodes_to_target.keys(), episodes_to_target.values())
    plt.xlabel('Algorithm')
    plt.ylabel(f'Episodes to reach target score of {target_score}')
    plt.title('Learning Efficiency Comparison')
    plt.grid(axis='y')
    
    # Add value labels on top of bars
    for algo, episodes in episodes_to_target.items():
        plt.text(algo, episodes + 5, str(episodes), ha='center')
    
    plt.tight_layout()
    plt.savefig('results/learning_efficiency.png')
    plt.show()

# Main function to train and compare all algorithms
def main():
    # List of algorithms to train
    algorithms = ['Sarsa', 'QLearning', 'NStepQLearning', 'A3C']
    
    # Train each algorithm and collect results
    results = {}
    for algo in algorithms:
        results[algo] = train_algorithm(algo)
    
    # Plot results
    plot_results(results)
    plot_learning_efficiency(results)
    
    # Final evaluation with more episodes
    print("\nFinal Evaluation (100 episodes each):")
    for algo, data in results.items():
        final_score = evaluate_model(data['model'], 100)
        print(f"{algo}: {final_score:.2f}")

if __name__ == "__main__":
    main()