# HASAC: Hyper-Actor Soft Actor-Critic

## Overview
"Adaptable automation with modular deep reinforcement learning and policy transfer" introduces the Hyper-Actor Soft Actor-Critic (HASAC) framework, designed to enhance the adaptability of collaborative robots in industrial settings. This deep Reinforcement Learning (RL) framework focuses on efficient and scalable training across a variety of tasks, leveraging task modularization and transfer learning for enhanced performance.

## Description
The HASAC framework is motivated by the need for intelligent, adaptable automation systems in future industrial environments. By integrating advanced deep RL algorithms, HASAC aims to provide autonomous learning capabilities to robots, enabling them to master various manipulation tasks with minimal human intervention.

### Key Features:
- **Modular Task Learning:** Facilitates the training on diverse tasks using MetaWorld's Sawyer robotic environments.
- **Policy Transfer:** Enhances adaptability to new tasks by transferring learned policies through a "hyper-actor".
- **Efficient and Stable:** Addresses the limitations of sample inefficiency and instability in traditional deep RL algorithms.
- **Robotic Manipulation Benchmarking:** Tested on the Meta-World virtual robotic manipulation benchmark.

## Usage
The HASAC framework can be integrated into robotic systems for various manipulation tasks. It involves training a deep RL agent using the HyperNet mechanism to effectively transfer policies and adapt to new tasks.

Example usage snippet:
```python
# For training
from rl_modules.AC_agent import AC_agent
from metaworld.benchmarks import ML1
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv

env = SawyerReachPushPickPlaceEnv()
args = get_args()  # Load or define training arguments
ac_agent = AC_agent(args, env, get_env_params(env), modules, HyperNet)
ac_agent.learn(task, HyperNet)

# For testing
env = SawyerButtonPressEnv()
env.set_parameters(obs_type='with_goal', random_init=True, task_type='reach')
obs = env.reset()
# Perform test actions with the trained model
