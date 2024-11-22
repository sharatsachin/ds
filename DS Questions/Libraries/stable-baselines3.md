# Stable Baselines 3

## What is Stable Baselines 3?

Stable Baselines 3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. Unlike its predecessor (Stable Baselines), SB3:
- Is built on PyTorch instead of TensorFlow
- Has cleaner code architecture
- Provides better typing
- Features improved documentation
- Offers better performance

## What are the core concepts in Stable Baselines 3?

The core concepts are:
1. **Environment**: The problem space (typically a Gym/Gymnasium environment)
2. **Agent**: The RL algorithm that learns to solve the environment
3. **Policy**: The strategy that the agent uses to take actions
4. **Observation Space**: The input state space
5. **Action Space**: The possible actions the agent can take

## What algorithms are available in Stable Baselines 3?

SB3 implements these key algorithms:
1. **A2C** (Advantage Actor Critic)
2. **DQN** (Deep Q Network)
3. **PPO** (Proximal Policy Optimization)
4. **SAC** (Soft Actor Critic)
5. **TD3** (Twin Delayed DDPG)
6. **DDPG** (Deep Deterministic Policy Gradient)

## How do you install Stable Baselines 3?

```bash
# Basic installation
pip install stable-baselines3

# With extra dependencies
pip install stable-baselines3[extra]

# With specific torch version
pip install torch==2.0.0 stable-baselines3
```

## How do you create and train a model?

Basic training example:
```python
from stable_baselines3 import PPO
import gymnasium as gym

# Create environment
env = gym.make("CartPole-v1")

# Initialize agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_cartpole")

# Load the model
loaded_model = PPO.load("ppo_cartpole")
```

## What are the common policies in Stable Baselines 3?

1. **MlpPolicy**: For vector inputs
```python
model = PPO("MlpPolicy", env)
```

2. **CnnPolicy**: For image inputs
```python
model = PPO("CnnPolicy", env)
```

3. **MultiInputPolicy**: For multiple input types
```python
model = PPO("MultiInputPolicy", env)
```

## How do you customize the policy network?

Example of custom network architecture:
```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
    def forward(self, observations):
        return self.cnn(observations)

# Use custom network
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
```

## How do you customize training parameters?

Common parameter customization:
```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/"
)
```

## How do you implement callbacks?

Common callback examples:
```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logs/",
    name_prefix="rl_model"
)

# Evaluation callback
eval_env = gym.make("CartPole-v1")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/results",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Use callbacks during training
model.learn(
    total_timesteps=10000,
    callback=[checkpoint_callback, eval_callback]
)
```

## How do you evaluate a trained model?

Evaluation examples:
```python
from stable_baselines3.common.evaluation import evaluate_policy

# Basic evaluation
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10,
    deterministic=True
)

# Custom evaluation loop
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## How do you handle custom environments?

Creating a custom environment:
```python
import gymnasium as gym
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
    
    def step(self, action):
        # Implement environment step
        observation = self._get_obs()
        reward = self._get_reward(action)
        done = self._is_done()
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        # Reset environment state
        return self._get_obs()

# Use custom environment
env = CustomEnv()
model = PPO("MlpPolicy", env)
```

## How do you implement vectorized environments?

Using vectorized environments:
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import make_vec_env

# Create vectorized environment
env = make_vec_env("CartPole-v1", n_envs=4)

# Or manually
def make_env():
    def _init():
        return gym.make("CartPole-v1")
    return _init

# Single process
env = DummyVecEnv([make_env() for _ in range(4)])

# Multi-process
env = SubprocVecEnv([make_env() for _ in range(4)])
```

## How do you handle logging and monitoring?

Setting up logging:
```python
from stable_baselines3.common.logger import configure

# Configure logger
new_logger = configure("./logs", ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Monitor training with Tensorboard
model = PPO("MlpPolicy", env, tensorboard_log="./tensorboard_logs/")

# Access training stats
model.logger.dir  # Get logging directory
```

## What are some best practices for using Stable Baselines 3?

1. **Environment Checking**:
```python
from stable_baselines3.common.env_checker import check_env

# Validate custom environment
check_env(env)
```

2. **Hyperparameter Optimization**:
```python
from stable_baselines3.common.utils import linear_schedule

# Learning rate schedule
learning_rate = linear_schedule(0.001, 0.0001)
model = PPO("MlpPolicy", env, learning_rate=learning_rate)
```

3. **Deterministic Training**:
```python
# Set seeds for reproducibility
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

model = PPO("MlpPolicy", env, seed=42)
```

## How do you handle action masks?

Implementing action masking:
```python
class MaskedEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(4,))
        })
    
    def _get_action_mask(self):
        # Return boolean mask of valid actions
        return np.array([1, 1, 0, 1])  # Action 2 is invalid