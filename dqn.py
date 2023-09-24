import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import gym

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        
        # First fully connected layer
        # input shape: (batch_size, input_dim)
        # output shape: (batch_size, hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Second fully connected layer
        # input shape: (batch_size, hidden_dim)
        # output shape: (batch_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Third fully connected layer that outputs Q-values
        # input shape: (batch_size, hidden_dim)
        # output shape: (batch_size, output_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass through first layer with ReLU activation
        # input shape: (batch_size, input_dim)
        # output shape: (batch_size, hidden_dim)
        x = torch.relu(self.fc1(x))
        
        # Pass through second layer with ReLU activation
        # input shape: (batch_size, hidden_dim)
        # output shape: (batch_size, hidden_dim)
        x = torch.relu(self.fc2(x))
        
        # Return the Q-values
        # input shape: (batch_size, hidden_dim)
        # output shape: (batch_size, output_dim)
        return self.fc3(x)

# Define the prioritized replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        # Memory to store experiences
        self.memory = deque(maxlen=capacity)
        # Store priorities for each experience
        self.priorities = deque(maxlen=capacity)
        # Named tuple to represent an experience
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        # Hyperparameters for prioritization
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        # Small value to ensure non-zero priority
        self.epsilon = 1e-5

    def push(self, *args):
        # Get the max priority or set to 1.0 if memory is empty
        max_priority = max(self.priorities) if self.memory else 1.0
        # Add experience to memory
        self.memory.append(self.Transition(*args))
        # Add priority for the experience
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        # Convert priorities to numpy array
        # shape: (len(self.memory),)
        priorities = np.array(self.priorities)
        # Calculate the probability of each experience being sampled
        # shape: (len(self.memory),)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample experiences based on their probabilities
        # shape: (batch_size,)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        # shape: (batch_size,)
        samples = [self.memory[idx] for idx in indices]

        # Calculate weights for each sampled experience
        # shape: (batch_size,)
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        # Increment beta towards 1
        self.beta = np.min([1., self.beta + self.beta_increment])

        # shape: (batch_size,), (batch_size,), (batch_size,)
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, errors):
        # Update priorities based on TD error
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon

    def __len__(self):
        return len(self.memory)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
BUFFER_SIZE = 10000
UPDATE_EVERY = 4
TAU = 0.001

# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Local Q-network, which will be trained
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        # Target Q-network, which will be used for stability
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        # Optimizer for training the local Q-network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Prioritized replay memory
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE)
        # Counter to track when to update the network
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.push(state, action, reward, next_state, done)
        # Increment the step counter
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # Check if it's time to update the network
        if self.t_step == 0:
            # Check if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # Sample experiences from memory
                experiences, indices, weights = self.memory.sample(BATCH_SIZE)
                # Learn from experiences
                self.learn(experiences, GAMMA, indices, weights)

    def act(self, state, eps=0.):
        # Convert state to tensor and send to device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # shape: (1, state_size)
        # Set the local Q-network to evaluation mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            # Get Q-values for the state
            action_values = self.qnetwork_local(state)  # shape: (1, action_size)
        # Set the local Q-network back to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, indices, weights):
        # Unpack experiences
        states, actions, rewards, next_states, dones = zip(*experiences)
        # Convert to tensors and send to device
        states = torch.from_numpy(np.vstack(states)).float().to(device)  # shape: (batch_size, state_size)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)  # shape: (batch_size, 1)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)  # shape: (batch_size, 1)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)  # shape: (batch_size, state_size)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)  # shape: (batch_size, 1)
        weights = torch.from_numpy(weights).float().to(device)  # shape: (batch_size,)

        # Get max predicted Q-values for next states from target model
        # note the call to .detach(), this prevents the target network from being trained
        # the target network is only updated using soft_update
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  # shape: (batch_size, 1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  # shape: (batch_size, 1)
        # Get expected Q-values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)  # shape: (batch_size, 1)

        # Compute TD error
        errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()  # shape: (batch_size, 1)
        # Update priorities in replay memory
        self.memory.update_priorities(indices, errors)

        # Compute loss using the weights from prioritized replay
        # this calculates the MSELoss between the rewards returned by each network (after adjusting Q_targets)
        # the idea is that over time the equation should result into minimum differences, which means an 
        # effective policy was learned. The optimizer will force the local network weights to adjust to minimize
        # the errors. The target network is considered a reference. Over time, we slowly move weights from the
        # local network into the target network (using soft_update below)
        loss = (torch.tensor(weights).to(device) * nn.MSELoss(reduction='none')(Q_expected, Q_targets)).mean()
        # Zero gradients
        self.optimizer.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Perform a step of optimization
        self.optimizer.step()

        # Softly update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        # Softly update target model parameters with local model parameters
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

if __name__ == "__main__":
    # Create CartPole environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # Initialize agent
    agent = DQN(state_size, action_size)

    n_episodes = 500
    max_t = 200
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    # Loop over episodes
    for episode in range(n_episodes):
        state = env.reset()
        # Loop over time steps
        for t in range(max_t):
            # Agent takes action
            action = agent.act(state, eps)
            # Environment returns next state and reward after action
            next_state, reward, done, _ = env.step(action)
            # Agent takes a step (stores experience and learns if necessary)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            # If episode is done, break
            if done:
                break
        # Decay epsilon for epsilon-greedy action selection
        eps = max(eps_end, eps_decay*eps)
        print(f"Episode {episode}/{n_episodes} | Length: {t+1}")
