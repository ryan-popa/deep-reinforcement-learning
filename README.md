# deep-reinforcement-learning

Clear implementations of some popular deep reinforcement learning algoriths with extensive comments.
This repo is used as supporting material for the training.
Feel free to contribute.

## Algorithms covered in the repo
- Double DQN (with prioritized experience replay).
- PPO (simplified, without batching. Check [here](https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py) a more complex pytorch implementation using advantage, actor-critic and batching from philtabor)
- A2C 
- REINFORCE
- DDPG``

## Reinforcement Learning Algorithm Classification

### 1. Based on Value vs. Policy:
**Value-Based**:
- Focus on learning the value function (either state value \(V(s)\) or action value \(Q(s, a)\)).
- If you look at dqn.py you can see our NN network takes as input the current environment state and predicts the value of taking each action in that state. For example for the cartpole environment we are predicting the value of moving left or right given the current state.
- Examples:
  - Q-Learning
  - Deep Q Networks (DQN)
  - Double DQN
  - Dueling DQN

**Policy-Based**:
- Directly optimize the policy without relying on a value function.
- If you look inside the PPO examples you can see the NN is predicting the actions that should be taken using Softmax. The NN is telling us directly what action we should take given any input state.
- Examples:
  - REINFORCE
  - Policy Gradient Methods
  - Trust Region Policy Optimization (TRPO)
  - Proximal Policy Optimization (PPO)

**Actor-Critic**:
- Combine both value-based and policy-based approaches.
- Examples:
  - Advantage Actor-Critic (A2C/A3C)
  - Deep Deterministic Policy Gradient (DDPG)
  - Soft Actor-Critic (SAC)
  - Twin Delayed Deep Deterministic Policy Gradient (TD3)

### 2. Based on Model Usage:
**Model-Free**:
- Learn directly from interaction without learning a model of the environment.
- Examples:
  - Q-Learning
  - DQN
  - REINFORCE
  - PPO
  - SAC

**Model-Based**:
- Learn a model of the environment and use it to make decisions.
- Examples:
  - Dyna-Q
  - PILCO
  - Model-Based Value Expansion (MBVE)

### 3. Based on Exploration Strategy:
**Epsilon-Greedy**:
- Use a probability \( \epsilon \) for exploration vs. exploitation.
- Examples:
  - Q-Learning
  - DQN

**Softmax Exploration**:
- Choose actions based on a probability distribution from action values.
- Examples:
  - Boltzmann Exploration in some policy gradient methods

**Entropy-Regularized Exploration**:
- Encourage exploration by adding an entropy term to the objective.
- Examples:
  - Soft Actor-Critic (SAC)

### 4. Based on On-Policy vs. Off-Policy:
**On-Policy**:
- Learn about the current policy being used.
- When you look into examples such as dpo.py and reinforce.py files you can see we are storing the entire episode and we are learning by comparing the new policy with the old policy.
- Examples:
  - REINFORCE
  - SARSA
  - A2C/A3C
  - TRPO
  - PPO

**Off-Policy**:
- Learn from past experiences regardless of the policy that generated them.
- If you look at dqn.py file you will see we are keeping all experiences in a buffer. Even though we 
- Examples:
  - Q-Learning
  - DQN
  - DDPG
  - SAC
  - TD3

### 5. Based on Discrete vs. Continuous Action Spaces:
**Discrete Action Algorithms**:
- Designed for finite action spaces.
- Examples:
  - Q-Learning
  - DQN

**Continuous Action Algorithms**:
- Designed for continuous action values.
- Examples:
  - DDPG
  - TRPO
  - PPO
  - SAC
  - TD3
