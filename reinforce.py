import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        # Pass the input through the layers with ReLU activations
        x = torch.relu(self.fc1(x))
        # The output layer uses softmax to produce a probability distribution over actions
        return torch.softmax(self.fc2(x), dim=1)  # shape: (batch_size, action_size)

class REINFORCE:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99):
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # shape: (1, state_size)
        probs = self.policy(state)  # shape: (1, action_size)
        action = np.random.choice(len(probs[0]), p=probs.detach().cpu().numpy()[0])  # scalar value
        return action

    def update(self, states, actions, rewards):
        states = torch.tensor(states, dtype=torch.float32).to(device)  # shape: (num_steps, state_size)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)  # shape: (num_steps,)

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards).to(device)  # shape: (num_steps,)
        
        # At this point we have the discounted rewards for each step in the trajectory
        # So for example:
        # step             : 0, 1, 2, 3, 4
        # action taken     : 1, 0, 1, 0, 1
        # rewards          : 3, 2, 1, 2, 3
        # discounted reward: 9.2, 6.2, 4, 2, 3

        # Normalize the rewards for stability
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Calculate the policy gradient loss
        log_probs = torch.log(self.policy(states).gather(1, actions.unsqueeze(1)).squeeze())  # shape: (num_steps,)
        # We use log for probabilities because it helps in numerical stability and also because of the properties of logarithms.
        # Multiplying with rewards is done to weight the actions that resulted in higher rewards more heavily during the optimization process.
        loss = -torch.sum(log_probs * discounted_rewards)  # scalar value

        # Backpropagate the loss and update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = REINFORCE(state_size, action_size)

    n_episodes = 500
    max_t = 200

    for episode in range(n_episodes):
        state = env.reset()
        episode_rewards = []
        states = []
        actions = []

        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            if done:
                break

        agent.update(states, actions, episode_rewards)
        print(f"Episode {episode}/{n_episodes} | Reward: {sum(episode_rewards)}")