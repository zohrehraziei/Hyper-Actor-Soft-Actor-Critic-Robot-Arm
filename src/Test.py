import torch
from rl_modules.models import actor
from arguments import get_args
import numpy as np
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press import SawyerButtonPressEnv

env = SawyerButtonPressEnv()
env.set_parameters(obs_type='with_goal', random_init=True, task_type='reach')
env.reset_model()
obs1 = env.reset()
obs = env.get_env_state()
#obs = env._get_obs()
print(obs)
print(obs1)