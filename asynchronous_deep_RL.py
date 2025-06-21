import gymnasium as gym
import numpy as np
import tensorflow as tf
import threading
import multiprocessing

# Global parameters
ENV_NAME = "CartPole-v1"
NUM_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EPISODES = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic
N_STEP_RETURN = 5  # for n-step algorithms

# Create global coordinator
global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
coordinator = tf.train.Coordinator()

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

# Base Worker class
class Worker(threading.Thread):
    def __init__(self, name, global_model, global_actor_optimizer, global_critic_optimizer):
        super(Worker, self).__init__()
        self.name = name
        self.global_model = global_model
        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer
        self.local_model = build_network(name)
        self.env = gym.make(ENV_NAME)
        
    def run(self):
        total_step = 1
        while not coordinator.should_stop() and global_episodes.numpy() < MAX_GLOBAL_EPISODES:
            # Sync with global network
            self.local_model.set_weights(self.global_model.get_weights())
            
            # Collect experience based on algorithm type
            self._collect_experience_and_update()
            
            if total_step % 100 == 0:
                print(f"Worker {self.name}, Episode: {global_episodes.numpy()}")
            
            total_step += 1
    
    def _collect_experience_and_update(self):
        # This method will be overridden by specific algorithm implementations
        pass

# Asynchronous One-step Sarsa Worker
class SarsaWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        buffer_s, buffer_a, buffer_r = [], [], []
        
        # Choose initial action using epsilon-greedy
        action_probs, _ = self.local_model(np.array([state]))
        action = np.random.choice(range(N_A), p=action_probs.numpy()[0])
        
        episode_reward = 0
        
        while not done:
            # Take action and observe
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Choose next action using epsilon-greedy
            next_action_probs, _ = self.local_model(np.array([next_state]))
            next_action = np.random.choice(range(N_A), p=next_action_probs.numpy()[0])
            
            # Store transition
            buffer_s.append(state)
            buffer_a.append(action)
            buffer_r.append(reward)
            
            # Calculate TD target for Sarsa: r + γQ(s',a')
            with tf.GradientTape() as tape:
                action_probs, value = self.local_model(np.array([state]))
                next_action_probs, next_value = self.local_model(np.array([next_state]))
                
                # Sarsa target uses the actual next action
                target = reward + (0 if done else GAMMA * next_value)
                advantage = target - value
                
                # Actor loss (policy gradient with advantage)
                action_oh = tf.one_hot(action, N_A, dtype=tf.float32)
                log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh))
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
            
        # Update global episode counter
        with tf.GradientTape() as tape:
            tf.assign_add(global_episodes, 1)

# Asynchronous One-step Q-Learning Worker
class QLearningWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Choose action using epsilon-greedy
            action_probs, _ = self.local_model(np.array([state]))
            action = np.random.choice(range(N_A), p=action_probs.numpy()[0])
            
            # Take action and observe
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Calculate TD target for Q-learning: r + γ max_a' Q(s',a')
            with tf.GradientTape() as tape:
                action_probs, value = self.local_model(np.array([state]))
                next_action_probs, next_value = self.local_model(np.array([next_state]))
                
                # Q-learning uses max Q-value for next state
                target = reward + (0 if done else GAMMA * next_value)
                advantage = target - value
                
                # Actor loss
                action_oh = tf.one_hot(action, N_A, dtype=tf.float32)
                log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh))
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
            
        # Update global episode counter
        with tf.GradientTape() as tape:
            tf.assign_add(global_episodes, 1)

# Asynchronous n-step Q-Learning Worker
class NStepQLearningWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        buffer_s, buffer_a, buffer_r = [], [], []
        episode_reward = 0
        
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
                action = np.random.choice(range(N_A), p=action_probs.numpy()[0])
                
                # Take action and observe
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
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
                    log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh))
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
        
        # Update global episode counter
        with tf.GradientTape() as tape:
            tf.assign_add(global_episodes, 1)

# Asynchronous Advantage Actor-Critic (A3C) Worker
class A3CWorker(Worker):
    def _collect_experience_and_update(self):
        state, _ = self.env.reset()
        done = False
        buffer_s, buffer_a, buffer_r = [], [], []
        episode_reward = 0
        
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
                action = np.random.choice(range(N_A), p=action_probs.numpy()[0])
                
                # Take action and observe
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
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
                    log_prob = tf.math.log(tf.reduce_sum(action_probs * action_oh))
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
        
        # Update global episode counter
        with tf.GradientTape() as tape:
            tf.assign_add(global_episodes, 1)

# Main function to create and run workers
def main(algorithm='A3C'):
    # Create global network
    global_model = build_network(GLOBAL_NET_SCOPE)
    
    # Create optimizers
    global_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_A)
    global_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_C)
    
    # Create workers based on algorithm choice
    workers = []
    for i in range(NUM_WORKERS):
        worker_name = f'{algorithm}_Worker_{i}'
        
        if algorithm == 'Sarsa':
            worker = SarsaWorker(worker_name, global_model, 
                                global_actor_optimizer, global_critic_optimizer)
        elif algorithm == 'QLearning':
            worker = QLearningWorker(worker_name, global_model, 
                                    global_actor_optimizer, global_critic_optimizer)
        elif algorithm == 'NStepQLearning':
            worker = NStepQLearningWorker(worker_name, global_model, 
                                        global_actor_optimizer, global_critic_optimizer)
        else:  # Default to A3C
            worker = A3CWorker(worker_name, global_model, 
                              global_actor_optimizer, global_critic_optimizer)
        
        workers.append(worker)
    
    # Start workers
    for worker in workers:
        worker.start()
    
    # Wait for all workers to finish
    coordinator.join(workers)

if __name__ == "__main__":
    # Choose algorithm: 'Sarsa', 'QLearning', 'NStepQLearning', or 'A3C'
    main(algorithm='A3C')

    ## For Asynchronous One-step Sarsa
    #main(algorithm='Sarsa')

    ## For Asynchronous One-step Q-Learning
    #main(algorithm='QLearning')

    ## For Asynchronous n-step Q-Learning
    #main(algorithm='NStepQLearning')

    ## For Asynchronous Advantage Actor-Critic (A3C)
    #main(algorithm='A3C')
