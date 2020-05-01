import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
from metaworld.benchmarks import ML1
from metaworld.envs.mujoco.sawyer_xyz import SawyerReachPushPickPlaceEnv


# process the inputs
def process_inputs(o, g):
    inputs = np.concatenate([o, g])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

# def compute_reward(achieved_goal, goal, info):
#     # Compute distance between goal and the achieved goal.
#     distance_threshold = 0.02
#     d = goal_distance(achieved_goal, goal)
#
#     return -(d > distance_threshold).astype(np.float32)
def compute_reward(achieved_goal, goal, maxReachDist):      ############################ Dense reward
    c1 = 1000
    c2 = 0.01
    c3 = 0.001
   # M = np.shape(achieved_goal)[0]
    maxReachDist = maxReachDist.flatten()
    reachDist = np.linalg.norm(achieved_goal - goal, axis=-1)
    # reachRew = -reachDist
    # if reachDist < 0.1:
    #     reachNearRew = 1000*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
    # else:
    #     reachNearRew = 0.
    reachRew = c1 * (maxReachDist - reachDist) + c1 * (np.exp(-(reachDist ** 2) / c2) + np.exp(-(reachDist ** 2) / c3))

    reachRew = np.maximum(reachRew, np.zeros(np.shape(reachRew)))
    #print(np.shape(reachRew))


    #print((maxReachDist - reachDist))
    # reachNearRew = max(reachNearRew,0)
    # reachRew = -reachDist
    reward = reachRew  # + reachNearRew
    #print(reward)
    return reward

def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    #print(g,g_mean)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': 3,
            'goal': 3,
            #'obs': obs['observation'].shape[0],
            #'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = 300
    return params

if __name__ == '__main__':
    args = get_args()
    # load the model param
    #model_path = args.save_dir + args.env_name + '/model.pt'
    #o_mean, o_std, g_mean, g_std, model, model1 = torch.load(model_path, map_location=lambda storage, loc: storage)
    #o_mean, o_std, g_mean, g_std, model, model1 = torch.load(model_path, map_location=lambda storage, loc: storage)

    # create the environment
    #env = ML1.get_train_tasks('push-v1')  # Create an environment with task `pick_place`
    env = SawyerReachPushPickPlaceEnv()
    #env.set_parameters(obs_type='with_goal', random_init=True, task_type='pick_place')
    env.obs_type = 'with_goal'
    env.task_type = 'reach'
    env.max_path_length = 3100
    env.reset_model()

    #print(env.observation_space)
    model_path = 'module_models/multitask.pt' 
    #model_actor = torch.load(model_path, map_location=lambda storage, loc: storage)
    o_mean, o_std, g_mean, g_std, model_actor, model_critic = torch.load(model_path, map_location=lambda storage, loc: storage)
    #print(ML1.available_tasks())  # Check out the available tasks

    # get the env param
    observation = env.reset()
    #print(observation)
    # get the environment params
    env_params = get_env_params(env)
    #print(observation)
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model_actor)
    #goal = np.array([0.1 , 0.8 , 0.02])
    for i in range(10):
        observation = env.reset()
        print(observation)
        # start to do the demo
        obs = observation[:3]
        g = observation[6:9]

        for t in range(3000):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            #inputs = np.concatenate([obs, g])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                #inputs = process_inputs(obs, g)
                pi = actor_network(inputs)
                actions = pi.detach().cpu().numpy().squeeze()
            actions = env.action_space.sample()  # Sample an action
            #put actions into the environment
            #ag = env.get_fingerCOM()

            observation_new, reward, _, info = env.step(actions)
            obs = observation_new[:3]
            success = info['success']
            #success = compute_reward(g, obs[:3], _) + 1
            #maxReachDist = env.get_maxReachDist()
            #print(observation_new[:3])
            #ag_new = env.get_fingerCOM()
            #REWARD = compute_reward(ag, ag_new, maxReachDist)
        env.close()
        print(observation_new)
        #print(reward)
        print(info)
        print('the episode is: {}'.format(i))
        print(success)
        #print(reward)
        #print(REWARD)

        #print(obs)
        #print(info)