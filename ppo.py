import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        # Pass the input through the layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # The output layer uses softmax to produce a probability distribution over actions
        return torch.softmax(self.fc3(x), dim=1)  # shape: (batch_size, action_size)

# Hyperparameters
LR = 0.002
GAMMA = 0.99
CLIP_EPSILON = 0.2
EPOCHS = 4
BATCH_SIZE = 32

class PPO:
    def __init__(self, state_size, action_size):
        # Define the current policy and the old policy (for trust region)
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.old_policy = PolicyNetwork(state_size, action_size).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer for updating the policy network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def select_action(self, state):
        # Convert the state to a tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # shape: (1, state_size)
        
        # Get the action probabilities from the policy network
        probs = self.policy(state)  # shape: (1, action_size)
        
        # Sample an action based on the probabilities
        action = np.random.choice(len(probs[0]), p=probs.detach().cpu().numpy()[0])  # shape: (1,)
        return action

    def update(self, states, actions, rewards, old_probs):
        # Convert lists to tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)  # shape: (num_steps, state_size)
        actions = torch.tensor(actions, dtype=torch.int32).to(device)  # shape: (num_steps,)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # shape: (num_steps,)
        old_probs = torch.tensor(old_probs, dtype=torch.float32).to(device)  # shape: (num_steps,)

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + GAMMA * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards).to(device)  # shape: (num_steps,)
        
        # Normalize the rewards for stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Optimize policy for EPOCHS epochs
        for _ in range(EPOCHS):
            # Get current action probabilities from the policy network
            current_probs = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # shape: (num_steps,)
            
            # Calculate the ratio of the new and old action probabilities
            ratio = current_probs / old_probs  # shape: (num_steps,)
            
            # Calculate the surrogate loss
            surrogate_loss = ratio * discounted_rewards  # shape: (num_steps,), values: (-inf, inf)
            
            # Clip the surrogate loss to ensure the new policy doesn't deviate too much from the old policy
            clipped_surrogate = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * discounted_rewards  # shape: (num_steps,), values: [1 - CLIP_EPSILON, 1 + CLIP_EPSILON] * (-inf, inf)
            
            # The final loss is the minimum of the surrogate loss and the clipped surrogate loss
            loss = -torch.min(surrogate_loss, clipped_surrogate).mean()  # scalar value, values: (-inf, inf)

            # Backpropagate the loss and update the policy network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update the old policy with the weights of the updated policy
        self.old_policy.load_state_dict(self.policy.state_dict())

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPO(state_size, action_size)

    n_episodes = 500
    max_t = 200

    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        states = []
        actions = []
        old_probs = []

        for t in range(max_t):
            action = agent.select_action(state)
            old_prob = agent.old_policy(torch.tensor(state, dtype=torch.float32).to(device))[0][action].item()  # scalar value
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            states.append(state)
            actions.append(action)
            old_probs.append(old_prob)
            state = next_state
            if done:
                break

        agent.update(states, actions, episode_rewards, old_probs)
        print(f"Episode {episode}/{n_episodes} | Reward: {sum(episode_rewards)}")