

import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
NUM_EPISODES = 1000

# Neural Network for Policy and Value estimation
# Both are based on the same neuronal network, which makes A2C both a value and policy based algorithm
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.actor = nn.Linear(128, n_actions)  # Outputs action probabilities
        self.critic = nn.Linear(128, 1)         # Outputs value estimate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

# Initialize environment and model
env = gym.make('CartPole-v1')
n_inputs = env.observation_space.shape[0]
n_outputs = env.action_space.n
model = ActorCritic(n_inputs, n_outputs)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for episode in range(NUM_EPISODES):
    done = False
    state = env.reset()
    episode_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: [1, n_inputs]

        # Get action probabilities and value estimate
        action_probs, value = model(state_tensor)  # action_probs shape: [1, n_outputs], value shape: [1, 1]

        # Sample action from the probabilities
        action = torch.multinomial(action_probs, 1).item()  # Scalar value

        # Take action in the environment
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Calculate advantage
        # Advantage is the difference between the expected return and the value estimate of the current state.
        # It is used to weight the actor and critic losses during optimization.
        # For example, if the advantage is positive, it means that the action taken in the current state resulted in a higher return than expected.
        # Therefore, the actor loss for that action should be increased, and the critic loss should be decreased.
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = model(next_state_tensor)
        if done:
            advantage = reward - value.item()
        else:
            advantage = reward + GAMMA * next_value.item() - value.item()
        
        # Calculate actor and critic losses
        # If the advantage is large it means we took an action which performed better than expected so we will increase it's probability in the future
        # The log is taken for numeric stability
        # The minus sign is because we want to maximize the advantage, but backpropagation will try to minimize the loss
        actor_loss = -torch.log(action_probs[0, action]) * advantage
        
        # The job of the critic is to correctly predict the value. The advantage represents an error for the critic. This is the equivalent of MSE
        critic_loss = advantage**2 
        
        # in order to prevent a local minimum for the loss we want want to penalize the optimization if it reduces the entropy. 
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-5))  # entropy is a measure of the policy's randomness, used to encourage exploration
        loss = actor_loss + critic_loss - ENTROPY_BETA * entropy  # the total loss is the sum of the actor and critic losses, minus the entropy multiplied by a hyperparameter

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

env.close()
