import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.AC_agent import AC_agent
import random
import torch
from rl_modules.models import actor, critic
from metaworld.benchmarks import ML1
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

tot_task = 3
tasks = ['reach','push','pick_place']
#tasks = ['push','pick_place']
modules = ['reach','push','pick_place']
task_module = [[0,0],[0,1],[0,2]] #This shows each task consists of which modules


class hyper_network():
    def __init__(self, args, env_params):
        # create the network
        self.actor_network = actor(env_params)
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=args.lr_actor)


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {
            'obs': 3,
            'goal': 3,
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'maxReachDist': 1
            }
    params['max_timesteps'] = 300
    return params

def launch(args):
    env = SawyerReachPushPickPlaceEnv()
    env_params = get_env_params(env)
    HyperNet = hyper_network(args,env_params)
    # create the ac_agent
    for task in range(tot_task):
        #task = 1
        env.obs_type = 'with_goal'
        env.random_init = True
        env.task_type = tasks[task]
        env.max_path_length = 700
        #env.set_parameters(obs_type='with_goal', random_init=True, task_type='reach')
        env.reset_model()
        ac_trainer = ac_agent(args, env, env_params,[modules[i] for i in task_module[task]],HyperNet)
        #ac_trainer.load_model(ac_trainer.model_path + '/model.pt')
        ac_trainer.learn(tasks[task],HyperNet)


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
